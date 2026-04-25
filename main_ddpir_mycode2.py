import argparse
import copy
import json
import logging
import os
import random
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils import utils_logger
from utils.mycode2_operators import build_operator


REPO_ROOT = Path(__file__).resolve().parent


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def repo_path(path_value: str | os.PathLike | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def str_to_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def find_nearest(array, value):
    if torch.is_tensor(array):
        return int(torch.abs(array.detach() - float(value)).argmin().item())
    array = np.asarray(array)
    return int(np.abs(array - value).argmin())


def diffpir_model_fn(
    x,
    noise_level,
    model_diffusion,
    vec_t=None,
    model_out_type="pred_xstart",
    diffusion=None,
    ddim_sample=False,
    alphas_cumprod=None,
    **model_kwargs,
):
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    reduced_alpha_cumprod = sqrt_1m_alphas_cumprod / sqrt_alphas_cumprod

    t_step = None
    if not torch.is_tensor(vec_t):
        t_step = find_nearest(reduced_alpha_cumprod, noise_level / 255.0)
        vec_t = torch.tensor([t_step] * x.shape[0], device=x.device)

    sampler = diffusion.ddim_sample if ddim_sample else diffusion.p_sample
    kwargs = dict(
        model=model_diffusion,
        x=x,
        t=vec_t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=model_kwargs,
    )
    if ddim_sample:
        kwargs["eta"] = 0
    out = sampler(**kwargs)

    if model_out_type == "pred_x_prev_and_start":
        return out["sample"], out["pred_xstart"]
    if model_out_type == "pred_x_prev":
        return out["sample"]
    if model_out_type == "pred_xstart":
        return out["pred_xstart"]
    if model_out_type in {"epsilon", "score"}:
        if t_step is None:
            t_step = int(vec_t[0].detach().item())
        alpha_prod_t = alphas_cumprod[int(t_step)]
        beta_prod_t = 1 - alpha_prod_t
        eps = (x - alpha_prod_t**0.5 * out["pred_xstart"]) / beta_prod_t**0.5
        return -eps / beta_prod_t**0.5 if model_out_type == "score" else eps
    raise ValueError(f"Unknown model_out_type: {model_out_type}")


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        image_root_path: Path,
        resolution: int,
        start_idx: int = 0,
        end_idx: int = -1,
        valid_extensions: Iterable[str] | None = None,
    ):
        if valid_extensions is None:
            valid_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        suffixes = {suffix.lower() for suffix in valid_extensions}
        self.fpaths = sorted(
            path
            for path in image_root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )
        if not self.fpaths:
            raise FileNotFoundError(f"No images found under {image_root_path}")
        self.fpaths = self.fpaths[start_idx:] if end_idx == -1 else self.fpaths[start_idx:end_idx]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(int(resolution)),
                transforms.CenterCrop(int(resolution)),
            ]
        )

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        image = Image.open(fpath).convert("RGB")
        tensor = self.transform(image) * 2.0 - 1.0
        return tensor, fpath.name


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_schedule(config: Dict[str, Any], device: torch.device):
    num_train_timesteps = int(config["num_train_timesteps"])
    beta_start = float(config["beta_start"])
    beta_end = float(config["beta_end"])
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    reduced_alpha_cumprod = sqrt_1m_alphas_cumprod / sqrt_alphas_cumprod
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_1m_alphas_cumprod": sqrt_1m_alphas_cumprod,
        "reduced_alpha_cumprod": reduced_alpha_cumprod,
    }


def make_sequence(config: Dict[str, Any]) -> List[int]:
    num_train_timesteps = int(config["num_train_timesteps"])
    iter_num = int(config["iter_num"])
    skip_type = config.get("skip_type", "quad")
    skip = num_train_timesteps // iter_num
    if skip_type == "uniform":
        seq = [i * skip for i in range(iter_num)]
        if skip > 1:
            seq.append(num_train_timesteps - 1)
    elif skip_type == "quad":
        seq = np.sqrt(np.linspace(0, num_train_timesteps**2, iter_num))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1
    else:
        raise ValueError(f"Unknown skip_type: {skip_type}")
    return seq


def compute_rhos(
    config: Dict[str, Any],
    schedule: Dict[str, torch.Tensor],
    sigma: float,
    device: torch.device,
):
    sqrt_alphas_cumprod = schedule["sqrt_alphas_cumprod"]
    sqrt_1m_alphas_cumprod = schedule["sqrt_1m_alphas_cumprod"]
    betas = schedule["betas"]
    alphas = schedule["alphas"]
    reduced_alpha_cumprod = schedule["reduced_alpha_cumprod"]
    model_out_type = config["model_output_type"]
    generate_mode = config["generate_mode"]

    sigmas = []
    sigma_ks = []
    rhos = []
    for i in range(int(config["num_train_timesteps"])):
        sigmas.append(reduced_alpha_cumprod[int(config["num_train_timesteps"]) - 1 - i])
        if model_out_type == "pred_xstart" and generate_mode == "DiffPIR":
            sigma_k = sqrt_1m_alphas_cumprod[i] / sqrt_alphas_cumprod[i]
        else:
            sigma_k = torch.sqrt(betas[i] / alphas[i])
        sigma_ks.append(sigma_k)
        rhos.append(float(config["lambda_"]) * (sigma**2) / (sigma_k**2))
    return (
        torch.tensor(rhos, device=device),
        torch.tensor(sigmas, device=device),
        torch.tensor(sigma_ks, device=device),
    )


def load_model(model_config: Dict[str, Any], device: torch.device):
    model_path = repo_path(model_config["model_path"])
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    guided_config = dict(
        model_path=str(model_path),
        image_size=int(model_config.get("image_size", 256)),
        num_channels=int(model_config.get("num_channels", 128)),
        num_res_blocks=int(model_config.get("num_res_blocks", 1)),
        channel_mult=str(model_config.get("channel_mult", "")),
        attention_resolutions=str(model_config.get("attention_resolutions", "16")),
        learn_sigma=bool(model_config.get("learn_sigma", True)),
        class_cond=bool(model_config.get("class_cond", False)),
        use_checkpoint=bool(model_config.get("use_checkpoint", False)),
        num_heads=int(model_config.get("num_heads", 4)),
        num_head_channels=int(model_config.get("num_head_channels", 64)),
        num_heads_upsample=int(model_config.get("num_heads_upsample", -1)),
        use_scale_shift_norm=bool(model_config.get("use_scale_shift_norm", True)),
        dropout=float(model_config.get("dropout", 0.0)),
        resblock_updown=bool(model_config.get("resblock_updown", True)),
        use_fp16=bool(model_config.get("use_fp16", False)),
        use_new_attention_order=bool(model_config.get("use_new_attention_order", False)),
    )
    diffusion_config = model_and_diffusion_defaults()
    diffusion_config.update(
        {key: value for key, value in guided_config.items() if key in diffusion_config}
    )
    model, diffusion = create_model_and_diffusion(**diffusion_config)
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
    return model.to(device), diffusion


def apply_data_consistency(
    x0: torch.Tensor,
    y: torch.Tensor,
    operator,
    rho: torch.Tensor,
    config: Dict[str, Any],
):
    mode = str(config.get("data_consistency", "auto"))
    tau = rho.float().clamp_min(float(config.get("rho_eps", 1e-12))).view(1, 1, 1, 1)

    if mode in {"auto", "analytic"} and hasattr(operator, "analytic_prox"):
        x0_p = operator.analytic_prox(x0, y, tau)
        return x0 + float(config.get("guidance_scale", 1.0)) * (x0_p - x0)
    if mode == "analytic":
        raise ValueError("Analytic data consistency was requested, but this operator has no prox.")

    x_work = x0
    step_scale = float(config.get("data_step_scale", 1.0))
    step_iters = int(config.get("data_step_iters", 1))
    reduction = str(config.get("data_loss_reduction", "mean"))
    clamp_after = bool(config.get("clamp_after_data", True))
    guidance_scale = float(config.get("guidance_scale", 1.0))

    for _ in range(step_iters):
        x_req = x_work.detach().requires_grad_(True)
        diff = operator(x_req) - y
        if reduction == "batch_sum":
            loss = 0.5 * diff.pow(2).flatten(1).sum(-1).mean()
        elif reduction == "sum":
            loss = 0.5 * diff.pow(2).sum()
        elif reduction == "mean":
            loss = 0.5 * diff.pow(2).mean()
        else:
            raise ValueError(f"Unknown data_loss_reduction: {reduction}")
        grad = torch.autograd.grad(loss, x_req)[0]
        x_candidate = x_req - step_scale * grad / tau
        x_work = x_req + guidance_scale * (x_candidate - x_req)
        if clamp_after:
            x_work = x_work.clamp(-1.0, 1.0)
    return x_work.detach()


def initialize_latent(
    images: torch.Tensor,
    y: torch.Tensor,
    operator,
    config: Dict[str, Any],
    schedule: Dict[str, torch.Tensor],
):
    reduced_alpha_cumprod = schedule["reduced_alpha_cumprod"]
    sqrt_alphas_cumprod = schedule["sqrt_alphas_cumprod"]
    sqrt_1m_alphas_cumprod = schedule["sqrt_1m_alphas_cumprod"]
    num_train_timesteps = int(config["num_train_timesteps"])

    if config.get("noise_init_img", "max") == "max":
        t_start = num_train_timesteps - 1
    else:
        t_start = find_nearest(
            reduced_alpha_cumprod,
            2 * float(config["noise_init_img"]) / 255.0,
        )

    if bool(config.get("init_from_measurement", True)):
        x_init = operator.initial_guess(y, tuple(images.shape)).to(images.device, images.dtype)
        t_y = find_nearest(reduced_alpha_cumprod, 2 * float(operator.sigma))
        sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
        noise_var = (
            sqrt_1m_alphas_cumprod[t_start] ** 2
            - sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y] ** 2
        ).clamp_min(0.0)
        x = sqrt_alpha_effective * x_init + torch.sqrt(noise_var) * torch.randn_like(x_init)
    else:
        x = torch.randn_like(images)
    return x, t_start


def run_diffpir_batch(
    images: torch.Tensor,
    y: torch.Tensor,
    operator,
    model,
    diffusion,
    config: Dict[str, Any],
    schedule: Dict[str, torch.Tensor],
):
    device = images.device
    x, t_start = initialize_latent(images, y, operator, config, schedule)
    sigma = max(0.001, float(operator.sigma))
    rhos, sigmas, _ = compute_rhos(config, schedule, sigma, device)
    seq = make_sequence(config)

    reduced_alpha_cumprod = schedule["reduced_alpha_cumprod"]
    sqrt_alphas_cumprod = schedule["sqrt_alphas_cumprod"]
    sqrt_1m_alphas_cumprod = schedule["sqrt_1m_alphas_cumprod"]
    betas = schedule["betas"]
    alphas_cumprod = schedule["alphas_cumprod"]

    x0 = x
    for seq_index, seq_value in enumerate(seq):
        curr_sigma = sigmas[seq_value].detach().cpu().numpy()
        t_i = find_nearest(reduced_alpha_cumprod, curr_sigma)
        if t_i > t_start:
            continue

        for inner_index in range(int(config["iter_num_U"])):
            with torch.no_grad():
                x0 = diffpir_model_fn(
                    x,
                    noise_level=curr_sigma * 255,
                    model_out_type=config["model_output_type"],
                    model_diffusion=model,
                    diffusion=diffusion,
                    ddim_sample=bool(config["ddim_sample"]),
                    alphas_cumprod=alphas_cumprod,
                )

            if seq_value != seq[-1]:
                x0 = apply_data_consistency(x0, y, operator, rhos[t_i], config)

            final_inner = inner_index == int(config["iter_num_U"]) - 1
            if config["model_output_type"] == "pred_xstart" and not (
                seq_value == seq[-1] and final_inner
            ):
                t_im1 = find_nearest(
                    reduced_alpha_cumprod,
                    sigmas[seq[seq_index + 1]].detach().cpu().numpy(),
                )
                eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                eta_sigma = (
                    float(config["eta"])
                    * sqrt_1m_alphas_cumprod[t_im1]
                    / sqrt_1m_alphas_cumprod[t_i]
                    * torch.sqrt(betas[t_i])
                )
                x = (
                    sqrt_alphas_cumprod[t_im1] * x0
                    + np.sqrt(1 - float(config["zeta"]))
                    * (
                        torch.sqrt(sqrt_1m_alphas_cumprod[t_im1] ** 2 - eta_sigma**2)
                        * eps
                        + eta_sigma * torch.randn_like(x)
                    )
                    + np.sqrt(float(config["zeta"]))
                    * sqrt_1m_alphas_cumprod[t_im1]
                    * torch.randn_like(x)
                )

            if inner_index < int(config["iter_num_U"]) - 1 and seq_value != seq[-1]:
                sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                noise_var = (
                    sqrt_1m_alphas_cumprod[t_i] ** 2
                    - sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1] ** 2
                ).clamp_min(0.0)
                x = sqrt_alpha_effective * x + torch.sqrt(noise_var) * torch.randn_like(x)

    sample_mode = str(config.get("final_sample", "xt"))
    sample = x0 if sample_mode == "x0" else x
    if hasattr(operator, "mask") and getattr(operator, "mask", None) is not None:
        mask = operator.mask.to(device=sample.device, dtype=torch.bool)
        sample = sample.clone()
        sample = torch.where(mask.expand_as(sample), y, sample)
    return sample.clamp(-1.0, 1.0)


def tensor_minus1_1_to_uint(tensor: torch.Tensor):
    img = ((tensor.detach().float().clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().numpy()
    if img.ndim == 4:
        img = np.transpose(img, (0, 2, 3, 1))
    return np.uint8((img * 255.0).round())


def save_tensor_batch(tensor: torch.Tensor, names: List[str], save_path: Path, prefix: str):
    save_path.mkdir(parents=True, exist_ok=True)
    imgs = tensor_minus1_1_to_uint(tensor)
    for index, name in enumerate(names):
        image = np.squeeze(imgs[index])
        if image.ndim == 2:
            Image.fromarray(image, mode="L").save(save_path / f"{prefix}{name}")
        else:
            Image.fromarray(image).save(save_path / f"{prefix}{name}")


def metrics_for_batch(
    samples: torch.Tensor,
    images: torch.Tensor,
    lpips_fn=None,
) -> Dict[str, float]:
    mse = torch.mean((samples - images) ** 2, dim=(1, 2, 3))
    psnr_values = 20 * torch.log10(2.0 / torch.sqrt(mse + 1e-10))
    result = {"psnr": float(psnr_values.mean().item())}
    if lpips_fn is not None:
        with torch.no_grad():
            lpips_value = lpips_fn(samples, images).mean().item()
        result["lpips"] = float(lpips_value)
    return result


def summarize_metrics(metric_totals: OrderedDict, total_images: int) -> OrderedDict:
    return OrderedDict(
        (name, value / max(1, total_images)) for name, value in metric_totals.items()
    )


def run_task(task_config: Dict[str, Any], dry_run: bool = False):
    seed = int(task_config["seed"])
    set_seed(seed)

    gpu = int(task_config.get("gpu", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{gpu}")
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    data_config = task_config["data"]
    image_root = repo_path(data_config["image_root_path"])
    if image_root is None:
        raise ValueError("data.image_root_path is required")
    total_images_override = task_config.get("total_images")
    end_idx = int(data_config.get("end_idx", -1))
    if total_images_override is not None:
        end_idx = int(data_config.get("start_idx", 0)) + int(total_images_override)
    dataset = ImageFolderDataset(
        image_root_path=image_root,
        resolution=int(data_config.get("resolution", 256)),
        start_idx=int(data_config.get("start_idx", 0)),
        end_idx=end_idx,
        valid_extensions=data_config.get("valid_extensions"),
    )

    task_name = str(task_config["task"])
    operator = build_operator(task_config["operator"], device)
    save_dir = repo_path(task_config.get("save_dir", "results/mycode2_inverse"))
    result_name = task_config.get("name", "DiffPIR_mycode2")
    output_path = save_dir / f"{result_name}_{task_name}_{task_config['operator']['name']}"

    model_path = repo_path(task_config["model"]["model_path"])
    if dry_run:
        print(
            json.dumps(
                {
                    "task": task_name,
                    "operator": task_config["operator"],
                    "dataset_images": len(dataset),
                    "image_root_path": str(image_root),
                    "model_path": str(model_path),
                    "model_exists": bool(model_path and model_path.exists()),
                    "output_path": str(output_path),
                    "device": str(device),
                },
                indent=2,
            )
        )
        return

    output_path.mkdir(parents=True, exist_ok=True)
    logger_name = f"{result_name}_{task_name}"
    utils_logger.logger_info(logger_name, log_path=str(output_path / f"{logger_name}.log"))
    logger = logging.getLogger(logger_name)
    logger.info("Resolved config:\n%s", yaml.safe_dump(task_config, sort_keys=False))

    shutil.copyfile(task_config["_source_config"], output_path / "task_config.yaml")
    with open(output_path / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(task_config, handle, sort_keys=False)

    dataloader = DataLoader(
        dataset,
        batch_size=int(task_config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model, diffusion = load_model(task_config["model"], device)
    schedule = make_schedule(task_config["diffpir"], device)

    lpips_fn = None
    if bool(task_config.get("calc_LPIPS", False)):
        import lpips

        lpips_fn = lpips.LPIPS(net=str(task_config.get("lpips_net", "vgg"))).to(device)

    metric_totals = OrderedDict()
    processed = 0
    for batch_index, (images, names) in enumerate(dataloader):
        images = images.to(device)
        names = list(names)
        y = operator.measure(images)
        samples = run_diffpir_batch(
            images,
            y,
            operator,
            model,
            diffusion,
            task_config["diffpir"],
            schedule,
        )
        batch_metrics = metrics_for_batch(samples, images, lpips_fn=lpips_fn)
        batch_size = images.shape[0]
        processed += batch_size
        for metric_name, metric_value in batch_metrics.items():
            metric_totals[metric_name] = metric_totals.get(metric_name, 0.0) + (
                metric_value * batch_size
            )
        logger.info(
            "batch %d/%d: %s",
            batch_index + 1,
            len(dataloader),
            ", ".join(f"{k}={v:.4f}" for k, v in batch_metrics.items()),
        )

        if bool(task_config.get("save_E", True)):
            save_tensor_batch(samples, names, output_path, "E_")
        if bool(task_config.get("save_L", True)):
            save_tensor_batch(operator.low_quality_image(y), names, output_path, "L_")
        if bool(task_config.get("save_H", False)):
            save_tensor_batch(images, names, output_path, "H_")

    summary = summarize_metrics(metric_totals, processed)
    with open(output_path / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("summary: %s", ", ".join(f"{k}={v:.4f}" for k, v in summary.items()))
    print(f"{task_name}: " + ", ".join(f"{k}={v:.4f}" for k, v in summary.items()))


def load_pipeline(pipeline_path: Path, selected_tasks: List[str] | None):
    with open(pipeline_path, "r", encoding="utf-8") as handle:
        pipeline = yaml.safe_load(handle)
    defaults = pipeline.get("defaults", {})
    task_entries = pipeline.get("tasks", [])
    selected = set(selected_tasks or [])
    loaded = []
    for entry in task_entries:
        if isinstance(entry, str):
            entry = {"name": entry, "config": f"configs/mycode2_inverse/{entry}.yaml"}
        entry_name = str(entry.get("name"))
        if selected and entry_name not in selected:
            continue
        task_path = repo_path(entry["config"])
        with open(task_path, "r", encoding="utf-8") as handle:
            task_config = yaml.safe_load(handle)
        merged = deep_merge(defaults, task_config)
        merged = deep_merge(
            merged,
            {k: v for k, v in entry.items() if k not in {"config", "name"}},
        )
        merged["_pipeline_name"] = entry_name
        merged["_source_config"] = str(task_path)
        loaded.append(merged)
    if selected and len(loaded) != len(selected):
        found = {config["_pipeline_name"] for config in loaded}
        missing = sorted(selected - found)
        raise ValueError(f"Unknown task selection: {missing}")
    return loaded


def apply_cli_overrides(task_configs: List[Dict[str, Any]], args):
    for config in task_configs:
        if args.total_images is not None:
            config["total_images"] = int(args.total_images)
        if args.batch_size is not None:
            config["batch_size"] = int(args.batch_size)
        if args.iter_num is not None:
            config["diffpir"]["iter_num"] = int(args.iter_num)
        if args.calc_lpips is not None:
            config["calc_LPIPS"] = bool(args.calc_lpips)
        if args.save_dir is not None:
            config["save_dir"] = args.save_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DiffPIR on the inverse-problem simulations used by mycode2."
    )
    parser.add_argument(
        "--pipeline",
        default="configs/mycode2_inverse_pipeline.yaml",
        help="Pipeline YAML listing the mycode2 inverse tasks.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Optional subset of task names from the pipeline.",
    )
    parser.add_argument("--total-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--iter-num", type=int, default=None)
    parser.add_argument("--calc-lpips", type=str_to_bool, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline_path = repo_path(args.pipeline)
    task_configs = load_pipeline(pipeline_path, args.tasks)
    apply_cli_overrides(task_configs, args)
    for task_config in task_configs:
        run_task(task_config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
