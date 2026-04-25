# mycode2 Inverse-Problem Pipeline for DiffPIR

This pipeline runs DiffPIR on the same FFHQ inverse-problem simulations used by
`C:/Users/Seif/Desktop/mycode2`.

## What Is Matched

The task configs in `configs/mycode2_inverse/` copy the operator settings from
`mycode2/configs/inverse_task/`:

- `down_sampling`: scale factor 4, resolution 256, sigma 0.05
- `inpainting_rand`: random mask probability range `[0.70, 0.71]`, sigma 0.05
- `motion_blur`: kernel size 61, intensity 0.5, sigma 0.05
- `gaussian_blur`: kernel size 61, intensity 3.0, sigma 0.05
- `inpainting_box`: box mask length range `[128, 129]`, sigma 0.05
- `phase_retrieval`: oversample 2.0, sigma 0.05

The default data/model settings point to:

- data: `../../mycode2/demo-samples/ffhq`
- model: `../../mycode2/pretrained-models/ffhq_10m.pt`

Paths are resolved relative to the DiffPIR repo root.

## Commands

Validate paths and task resolution without loading the model:

```powershell
python main_ddpir_mycode2.py --dry-run
```

Run all six tasks:

```powershell
python main_ddpir_mycode2.py
```

Run a subset:

```powershell
python main_ddpir_mycode2.py --tasks down_sampling gaussian_blur phase_retrieval
```

Quick smoke run:

```powershell
python main_ddpir_mycode2.py --tasks inpainting_box --total-images 1 --batch-size 1 --iter-num 1 --calc-lpips false
```

Outputs are written under `results/mycode2_inverse/` by default.

## Colab

The Colab notebook is generated at:

```text
notebooks/DiffPIR_mycode2_inverse_colab.ipynb
```

Open it in Colab and run the cells. The notebook defaults to
`https://github.com/Seif-Hussein/DiffPIR.git` on branch
`codex-diffpir-mycode2-colab`, downloads `ffhq_10m.pt`, writes a Colab-specific
pipeline under `configs/colab_mycode2_inverse/`, dry-runs the selected tasks,
then launches the simulations.

The Colab defaults now mirror the PDHG single-run notebooks:

- Drive dataset: `/content/drive/MyDrive/mycode/test-ffhq`
- image slice: `DATA_START_IDX=0`, `TOTAL_IMAGES=100`
- requested batch size: `BATCH_SIZE=100`
- measurement noise: `sigma=0.05` for every inverse problem

For speed, the notebook copies the selected Drive images into
`/content/diffpir_test_ffhq_cache` once and runs DiffPIR from that local runtime
cache. The runner also tries the requested batch size first and automatically
splits a batch if CUDA reports out-of-memory, so the Colab uses the largest
working batch size instead of failing immediately.

Each task output folder also gets live run-state JSON files:

- `progress.json`: overwritten throughout the run with status, processed image
  counts, percent complete, overall elapsed seconds per image, processing-only
  seconds per image, ETA, and current metrics
- `history.json`: cumulative per-chunk history, including filenames, timings,
  running metrics, and OOM split events
- `metric_history.json`: same history payload under the metric-history name used
  by the PDHG notebooks

## Paper Hyperparameters

For the local configs and the Colab notebook, all operators use `sigma=0.05`.
DiffPIR hyperparameters use the paper's FFHQ/NFE=100 values where reported:

- `down_sampling` / SR x4: `lambda_=8.0`, `zeta=0.2`
- `gaussian_blur`: `lambda_=7.0`, `zeta=0.3`
- `motion_blur`: `lambda_=7.0`, `zeta=0.4`
- `inpainting_box`: `lambda_=6.0`, `zeta=0.5`
- `inpainting_rand`: `lambda_=7.0`, `zeta=1.0`

DiffPIR does not report a phase-retrieval preset. The notebook and local config
use the inverse-problem operator default `oversample=2.0` and a conservative
first-order DiffPIR starter preset, `lambda_=8.0`, `zeta=0.3`.

## DiffPIR Data Consistency

DiffPIR's original repo has analytic updates for its own SR/deblur/inpainting
degradations. Those are not always the same as `mycode2`:

- `mycode2` blur uses reflection padding, not DiffPIR's circular FFT blur.
- `mycode2` downsampling uses its `Resizer` operator.
- phase retrieval is nonlinear and has no closed-form DiffPIR update here.

To keep the measurement operators faithful to `mycode2`, this runner uses:

- analytic prox only for inpainting
- gradient data-consistency for downsampling, blur, and phase retrieval

The relevant knobs are in each task YAML under `diffpir`:

- `data_consistency`
- `data_step_scale`
- `data_step_iters`
- `data_loss_reduction`
- `guidance_scale`
