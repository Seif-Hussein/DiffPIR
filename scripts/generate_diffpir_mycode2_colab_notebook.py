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

        # Use DiffPIR's bundled demo images by default. You can point this to
        # a mounted Drive folder containing your FFHQ/mycode2 images.
        DATA_ROOT = "testsets/demo_test"  #@param {type:"string"}
        TOTAL_IMAGES = 3  #@param {type:"integer"}
        BATCH_SIZE = 1  #@param {type:"integer"}

        TASKS = [
            "down_sampling",
            "inpainting_rand",
            "motion_blur",
            "gaussian_blur",
            "inpainting_box",
            "phase_retrieval",
        ]

        CALC_LPIPS = False  #@param {type:"boolean"}
        RUN_INSTALL = True  #@param {type:"boolean"}
        DOWNLOAD_FFHQ_CHECKPOINT = True  #@param {type:"boolean"}
        """
    ),
    code(
        """
        #@title Clone or enter repository
        from pathlib import Path
        import os
        import subprocess
        import sys

        workdir = Path(WORKDIR)
        if REPO_URL:
            if workdir.exists():
                subprocess.run(["rm", "-rf", str(workdir)], check=True)
            subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(workdir)], check=True)

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
        #@title Optional: mount Google Drive for custom data
        # Uncomment these lines if DATA_ROOT should point into your Drive.
        #
        # from google.colab import drive
        # drive.mount('/content/drive')
        # DATA_ROOT = '/content/drive/MyDrive/path/to/images'
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
            "data_loss_reduction": "mean",
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
                "total_images": int(TOTAL_IMAGES),
                "batch_size": int(BATCH_SIZE),
                "save_dir": "results/colab_mycode2_inverse",
                "save_E": True,
                "save_L": True,
                "save_H": False,
                "calc_LPIPS": bool(CALC_LPIPS),
                "lpips_net": "vgg",
                "data": {
                    "name": "FFHQ",
                    "resolution": 256,
                    "image_root_path": str(DATA_ROOT),
                    "start_idx": 0,
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
            str(TOTAL_IMAGES),
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
        #@title Run selected simulations
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
            str(TOTAL_IMAGES),
            "--batch-size",
            str(BATCH_SIZE),
            "--calc-lpips",
            str(CALC_LPIPS).lower(),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
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
        for metrics_path in sorted(Path("results/colab_mycode2_inverse").glob("*/metrics.json")):
            metrics = json.loads(metrics_path.read_text())
            rows.append({"run": metrics_path.parent.name, **metrics})
        display(pd.DataFrame(rows))

        preview_paths = sorted(Path("results/colab_mycode2_inverse").glob("*/E_*.png"))[:8]
        for path in preview_paths:
            print(path)
            display(Image.open(path).resize((160, 160)))
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
