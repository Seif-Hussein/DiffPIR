import json
from textwrap import dedent
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "DiffPIR_mycode2_inverse_colab.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    md(
        """
        # DiffPIR Colab: mycode2-style inverse-problem simulations

        This notebook runs the six inverse-problem simulations you asked for with the DiffPIR runner in this repo:
        downsampling, random inpainting, motion blur, Gaussian blur, box inpainting, and phase retrieval.

        Source notes used for the presets:

        - DiffPIR paper: `2305.08995v1`, especially Appendix B / Table 3 for the paper's `lambda_` and `zeta` values.
        - DiffPIR repository: https://github.com/yuanzhi-zhu/DiffPIR for the public runner/config defaults, NFE=100 setting, quadratic sampling sequence, and checkpoint setup.
        - DAPS/DPS-style inverse-problem repositories: https://github.com/zhangbingliang2019/DAPS and https://github.com/DPS2022/diffusion-posterior-sampling for the broader inverse-problem operator defaults, including phase retrieval.

        Important caveat: DiffPIR reports paper defaults for SR/deblur/inpainting, but not for phase retrieval. Phase retrieval below uses the operator default from the inverse-problem repo lineage, plus a conservative DiffPIR first-order-prox starting preset.
        """
    ),
    md(
        """
        ## Presets encoded here

        | Notebook task | Operator setting | DiffPIR preset |
        |---|---:|---:|
        | `down_sampling` | x4 downsampling, sigma=0.05 | NFE=100, `lambda_=8.0`, `zeta=0.2` |
        | `inpainting_rand` | random mask, 70%-71% masked | NFE=100, `lambda_=7.0`, `zeta=1.0` |
        | `motion_blur` | kernel size 61, intensity 0.5, sigma=0.05 | NFE=100, `lambda_=7.0`, `zeta=0.4` |
        | `gaussian_blur` | kernel size 61, intensity 3.0, sigma=0.05 | NFE=100, `lambda_=7.0`, `zeta=0.3` |
        | `inpainting_box` | 128x128 box mask | NFE=100, `lambda_=6.0`, `zeta=0.5` |
        | `phase_retrieval` | oversample 2.0, sigma=0.05 | NFE=100, `lambda_=8.0`, `zeta=0.3` starter preset |

        All six inverse problems use measurement noise `sigma=0.05`, matching the setting for these simulations.
        """
    ),
    code(
        """
        #@title Colab controls
        # This branch contains main_ddpir_mycode2.py and the mycode2 inverse-problem presets.
        # If the repository is already present at WORKDIR, leave REPO_URL empty.
        REPO_URL = "https://github.com/Seif-Hussein/DiffPIR.git"  #@param {type:"string"}
        BRANCH = "codex-diffpir-mycode2-colab"  #@param {type:"string"}
        WORKDIR = "/content/DiffPIR"  #@param {type:"string"}

        # Matches the PDHG single-run notebooks.
        MOUNT_DRIVE = True  #@param {type:"boolean"}
        DRIVE_FFHQ_DATA_DIR = "/content/drive/MyDrive/mycode/test-ffhq"  #@param {type:"string"}
        CACHE_DATASET_TO_LOCAL = True  #@param {type:"boolean"}
        LOCAL_DATA_CACHE_DIR = "/content/diffpir_test_ffhq_cache"  #@param {type:"string"}
        DATA_START_IDX = 0  #@param {type:"integer"}
        TOTAL_IMAGES = 100  #@param {type:"integer"}
        BATCH_SIZE = 100  #@param {type:"integer"}

        TASKS = [
            "down_sampling",
            "inpainting_rand",
            "motion_blur",
            "gaussian_blur",
            "inpainting_box",
            "phase_retrieval",
        ]

        CALC_LPIPS = False  #@param {type:"boolean"}
        SAVE_E = True  #@param {type:"boolean"}
        SAVE_L = False  #@param {type:"boolean"}
        SAVE_H = False  #@param {type:"boolean"}
        RUN_INSTALL = True  #@param {type:"boolean"}
        DOWNLOAD_FFHQ_CHECKPOINT = True  #@param {type:"boolean"}
        RUN_IN_BACKGROUND = True  #@param {type:"boolean"}
        LOG_TAIL_LINES = 120  #@param {type:"integer"}
        DRIVE_EXPORT_DIR = "/content/drive/MyDrive/diffpir_mycode2_inverse_exports"  #@param {type:"string"}
        """
    ),
    code(
        """
        #@title Mount Google Drive
        if MOUNT_DRIVE:
            from google.colab import drive
            drive.mount("/content/drive")
        """
    ),
    code(
        """
        #@title Clone or enter repository
        from pathlib import Path
        import os
        import shutil
        import subprocess
        import sys

        workdir = Path(WORKDIR)
        if REPO_URL:
            if workdir.exists():
                shutil.rmtree(workdir)
            subprocess.run(
                ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(workdir)],
                check=True,
            )

        if not (workdir / "main_ddpir_mycode2.py").exists():
            raise RuntimeError(
                "main_ddpir_mycode2.py was not found. Set REPO_URL to your fork/branch "
                "or upload this repo to /content/DiffPIR."
            )

        os.chdir(workdir)
        print("Using repo:", Path.cwd())
        """
    ),
    code(
        """
        #@title Install lightweight Colab dependencies
        import subprocess
        import sys

        if RUN_INSTALL:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "gdown",
                    "lpips==0.1.4",
                    "blobfile",
                    "hdf5storage",
                    "pyyaml",
                ],
                check=True,
            )
        """
    ),
    code(
        """
        #@title Download FFHQ checkpoint
        from pathlib import Path
        import subprocess
        import sys

        ckpt = Path("pretrained-models/ffhq_10m.pt")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        if DOWNLOAD_FFHQ_CHECKPOINT and not ckpt.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gdown",
                    "https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh",
                    "-O",
                    str(ckpt),
                ],
                check=True,
            )
        print("checkpoint:", ckpt, "exists=", ckpt.exists(), "size=", ckpt.stat().st_size if ckpt.exists() else 0)
        """
    ),
    code(
        """
        #@title Prepare the FFHQ dataset slice
        from pathlib import Path
        import shutil

        valid_extensions = {".jpg", ".jpeg", ".png"}
        source_data_dir = Path(DRIVE_FFHQ_DATA_DIR)
        if not source_data_dir.exists():
            raise FileNotFoundError(f"FFHQ dataset path not found: {source_data_dir}")

        source_images = sorted(
            path
            for path in source_data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in valid_extensions
        )
        data_end_idx = int(DATA_START_IDX) + int(TOTAL_IMAGES)
        selected_images = source_images[int(DATA_START_IDX):data_end_idx]
        if not selected_images:
            raise FileNotFoundError(
                f"No FFHQ images found in slice [{DATA_START_IDX}, {data_end_idx}) under {source_data_dir}"
            )

        if len(selected_images) < int(TOTAL_IMAGES):
            print(f"Requested {TOTAL_IMAGES} images, found {len(selected_images)} in the selected slice.")

        if CACHE_DATASET_TO_LOCAL:
            effective_data_root = Path(LOCAL_DATA_CACHE_DIR)
            if effective_data_root.exists():
                shutil.rmtree(effective_data_root)
            effective_data_root.mkdir(parents=True, exist_ok=True)
            for index, src in enumerate(selected_images):
                shutil.copy2(src, effective_data_root / f"{index:05d}_{src.name}")
            effective_start_idx = 0
            print(f"Copied {len(selected_images)} images to local runtime cache: {effective_data_root}")
        else:
            effective_data_root = source_data_dir
            effective_start_idx = int(DATA_START_IDX)

        effective_total_images = len(selected_images)
        print(f"Dataset source: {source_data_dir}")
        print(f"Dataset used by DiffPIR: {effective_data_root}")
        print(f"Dataset slice: [{DATA_START_IDX}, {data_end_idx}) -> {effective_total_images} images")
        """
    ),
    code(
        """
        #@title Write Colab-specific pipeline configs
        from pathlib import Path
        import yaml

        measurement_sigma = 0.05
        task_dir = Path("configs/colab_mycode2_inverse")
        task_dir.mkdir(parents=True, exist_ok=True)

        common_diffpir = {
            "num_train_timesteps": 1000,
            "iter_num": 100,
            "iter_num_U": 1,
            "lambda_": 1.0,
            "zeta": 0.1,
            "eta": 0.0,
            "guidance_scale": 1.0,
            "generate_mode": "DiffPIR",
            "model_output_type": "pred_xstart",
            "ddim_sample": False,
            "skip_type": "quad",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "noise_init_img": "max",
            "init_from_measurement": True,
            "data_consistency": "auto",
            "data_step_iters": 1,
            "data_step_scale": 1.0,
            "data_loss_reduction": "sum",
            "clamp_after_data": True,
            "final_sample": "xt",
        }

        task_configs = {
            "down_sampling": {
                "task": "down_sampling",
                "operator": {
                    "name": "down_sampling",
                    "resolution": 256,
                    "channels": 3,
                    "scale_factor": 4,
                    "sigma": 0.05,
                },
                "diffpir": {
                    "lambda_": 8.0,
                    "zeta": 0.2,
                    "data_consistency": "gradient",
                    "data_loss_reduction": "sum",
                },
            },
            "inpainting_rand": {
                "task": "inpainting_rand",
                "operator": {
                    "name": "inpainting",
                    "mask_type": "random",
                    "mask_prob_range": [0.70, 0.71],
                    "resolution": 256,
                    "sigma": measurement_sigma,
                },
                "diffpir": {"lambda_": 7.0, "zeta": 1.0, "data_consistency": "auto"},
            },
            "motion_blur": {
                "task": "motion_blur",
                "operator": {
                    "name": "motion_blur",
                    "kernel_size": 61,
                    "intensity": 0.5,
                    "sigma": 0.05,
                },
                "diffpir": {
                    "lambda_": 7.0,
                    "zeta": 0.4,
                    "data_consistency": "gradient",
                    "data_loss_reduction": "sum",
                },
            },
            "gaussian_blur": {
                "task": "gaussian_blur",
                "operator": {
                    "name": "gaussian_blur",
                    "kernel_size": 61,
                    "intensity": 3.0,
                    "sigma": 0.05,
                },
                "diffpir": {
                    "lambda_": 7.0,
                    "zeta": 0.3,
                    "data_consistency": "gradient",
                    "data_loss_reduction": "sum",
                },
            },
            "inpainting_box": {
                "task": "inpainting_box",
                "operator": {
                    "name": "inpainting",
                    "mask_type": "box",
                    "mask_len_range": [128, 129],
                    "resolution": 256,
                    "sigma": measurement_sigma,
                },
                "diffpir": {"lambda_": 6.0, "zeta": 0.5, "data_consistency": "auto"},
            },
            "phase_retrieval": {
                "task": "phase_retrieval",
                "operator": {
                    "name": "phase_retrieval",
                    "oversample": 2.0,
                    "resolution": 256,
                    "sigma": 0.05,
                },
                "diffpir": {
                    "lambda_": 8.0,
                    "zeta": 0.3,
                    "init_from_measurement": False,
                    "data_consistency": "gradient",
                    "data_loss_reduction": "sum",
                    "data_step_scale": 0.5,
                    "final_sample": "x0",
                },
            },
        }

        for name, cfg in task_configs.items():
            (task_dir / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

        pipeline = {
            "defaults": {
                "seed": 99,
                "gpu": 0,
                "name": "DiffPIR_colab",
                "total_images": int(effective_total_images),
                "batch_size": int(BATCH_SIZE),
                "save_dir": "results/colab_mycode2_inverse",
                "save_E": bool(SAVE_E),
                "save_L": bool(SAVE_L),
                "save_H": bool(SAVE_H),
                "calc_LPIPS": bool(CALC_LPIPS),
                "lpips_net": "vgg",
                "data": {
                    "name": "FFHQ",
                    "resolution": 256,
                    "image_root_path": str(effective_data_root),
                    "start_idx": int(effective_start_idx),
                    "end_idx": -1,
                    "valid_extensions": [".jpg", ".jpeg", ".png"],
                },
                "model": {
                    "name": "ddpm",
                    "model_path": str(ckpt),
                    "image_size": 256,
                    "num_channels": 128,
                    "num_res_blocks": 1,
                    "channel_mult": "",
                    "attention_resolutions": "16",
                    "learn_sigma": True,
                    "class_cond": False,
                    "use_checkpoint": False,
                    "num_heads": 4,
                    "num_head_channels": 64,
                    "num_heads_upsample": -1,
                    "use_scale_shift_norm": True,
                    "dropout": 0.0,
                    "resblock_updown": True,
                    "use_fp16": False,
                    "use_new_attention_order": False,
                },
                "diffpir": common_diffpir,
            },
            "tasks": [
                {"name": name, "config": str(task_dir / f"{name}.yaml")}
                for name in task_configs
            ],
        }
        pipeline_path = Path("configs/colab_mycode2_inverse_pipeline.yaml")
        pipeline_path.write_text(yaml.safe_dump(pipeline, sort_keys=False))
        print(pipeline_path)
        """
    ),
    code(
        """
        #@title Dry-run resolved tasks
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "main_ddpir_mycode2.py",
            "--pipeline",
            "configs/colab_mycode2_inverse_pipeline.yaml",
            "--tasks",
            *TASKS,
            "--total-images",
            str(effective_total_images),
            "--batch-size",
            str(BATCH_SIZE),
            "--calc-lpips",
            str(CALC_LPIPS).lower(),
            "--dry-run",
        ]
        subprocess.run(cmd, check=True)
        """
    ),
    code(
        """
        #@title Build run command
        import json
        import shlex
        import subprocess
        import sys
        import time
        from pathlib import Path

        session_tag = time.strftime("%Y%m%d-%H%M%S")
        run_tag = f"diffpir_mycode2_{session_tag}"
        save_root = Path("results/colab_mycode2_inverse")
        run_aux_root = Path("single_runs")
        run_aux_root.mkdir(parents=True, exist_ok=True)
        latest_log_path = run_aux_root / f"{run_tag}.log"
        latest_pid_path = run_aux_root / f"{run_tag}.pid"

        run_cmd = [
            sys.executable,
            "main_ddpir_mycode2.py",
            "--pipeline",
            "configs/colab_mycode2_inverse_pipeline.yaml",
            "--tasks",
            *TASKS,
            "--total-images",
            str(effective_total_images),
            "--batch-size",
            str(BATCH_SIZE),
            "--calc-lpips",
            str(CALC_LPIPS).lower(),
        ]

        context_path = run_aux_root / f"{run_tag}.context.json"
        last_context = {
            "run_tag": run_tag,
            "save_root": save_root.as_posix(),
            "latest_log_path": latest_log_path.as_posix(),
            "latest_pid_path": latest_pid_path.as_posix(),
            "context_path": context_path.as_posix(),
            "run_cmd": run_cmd,
        }
        with context_path.open("w", encoding="utf-8") as handle:
            json.dump(last_context, handle, indent=2)

        print(f"Run tag: {run_tag}")
        print(f"Dataset: {effective_data_root}")
        print(f"Images: {effective_total_images}")
        print(f"Requested batch size: {BATCH_SIZE}")
        print(f"Log: {latest_log_path}")
        print("\\nCommand:\\n")
        print(" ".join(shlex.quote(part) for part in run_cmd))
        """
    ),
    code(
        """
        #@title Launch selected simulations
        import subprocess
        from pathlib import Path

        latest_log_path = Path(last_context["latest_log_path"])
        latest_pid_path = Path(last_context["latest_pid_path"])
        latest_log_path.parent.mkdir(parents=True, exist_ok=True)

        if RUN_IN_BACKGROUND:
            with latest_log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    last_context["run_cmd"],
                    cwd=Path.cwd(),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            latest_pid_path.write_text(str(process.pid), encoding="utf-8")
            print(f"PID: {process.pid}")
            print(f"Log: {latest_log_path}")
            print(f"Save root: {last_context['save_root']}")
        else:
            subprocess.run(last_context["run_cmd"], check=True)
        """
    ),
    code(
        """
        #@title Show recent log lines
        from pathlib import Path

        log_path = Path(last_context["latest_log_path"])
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = lines[-int(LOG_TAIL_LINES):]
        print("\\n".join(tail) if tail else "<log is empty>")
        """
    ),
    code(
        """
        #@title Show progress and timing history
        from pathlib import Path
        import json
        import pandas as pd
        from IPython.display import display

        save_root = Path(last_context["save_root"])

        progress_rows = []
        for progress_path in sorted(save_root.glob("*/progress.json")):
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress_rows.append(
                {
                    "run": progress_path.parent.name,
                    "status": progress.get("status"),
                    "processed": progress.get("processed_images"),
                    "total": progress.get("total_images"),
                    "percent": progress.get("percent_complete"),
                    "overall seconds/image": progress.get("elapsed_seconds_per_image"),
                    "processing seconds/image": progress.get("processing_elapsed_seconds_per_image"),
                    "eta_seconds": progress.get("estimated_remaining_seconds"),
                    "processing_eta_seconds": progress.get("estimated_processing_remaining_seconds"),
                    "updated_at": progress.get("updated_at"),
                }
            )

        display(pd.DataFrame(progress_rows))

        history_rows = []
        for history_path in sorted(save_root.glob("*/history.json")):
            history = json.loads(history_path.read_text(encoding="utf-8"))
            chunks = history.get("chunks", [])
            last_chunk = chunks[-1] if chunks else {}
            history_rows.append(
                {
                    "run": history_path.parent.name,
                    "status": history.get("status"),
                    "chunks": len(chunks),
                    "overall seconds/image": history.get("elapsed_seconds_per_image"),
                    "processing seconds/image": history.get("processing_elapsed_seconds_per_image"),
                    "last chunk seconds/image": last_chunk.get("elapsed_seconds_per_image"),
                    "processed": history.get("processed_images"),
                    "total": history.get("total_images"),
                    "oom retries": len(history.get("oom_retries", [])),
                }
            )

        display(pd.DataFrame(history_rows))

        for progress_path in sorted(save_root.glob("*/progress.json")):
            print(f"\\n{progress_path}")
            print(progress_path.read_text(encoding="utf-8"))
        """
    ),
    code(
        """
        #@title Summarize metrics and preview outputs
        from pathlib import Path
        import json
        import pandas as pd
        from IPython.display import display
        from PIL import Image

        rows = []
        save_root = Path(last_context["save_root"])
        for metrics_path in sorted(save_root.glob("*/metrics.json")):
            metrics = json.loads(metrics_path.read_text())
            rows.append({"run": metrics_path.parent.name, **metrics})
        display(pd.DataFrame(rows))

        preview_paths = sorted(save_root.glob("*/E_*.png"))[:8]
        for path in preview_paths:
            print(path)
            display(Image.open(path).resize((160, 160)))
        """
    ),
    code(
        """
        #@title Copy run artifacts to Drive
        import shutil
        from pathlib import Path

        export_root = Path(DRIVE_EXPORT_DIR)
        export_root.mkdir(parents=True, exist_ok=True)

        targets = [
            Path(last_context["save_root"]),
            Path(last_context["latest_log_path"]),
            Path(last_context["context_path"]),
        ]

        for src in targets:
            if not src.exists():
                print(f"Skipping missing path: {src}")
                continue

            dst = export_root / src.name
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(NOTEBOOK_PATH)
