# 🎯 S07 Assignment: Hyperparameter Optimization with GitHub Actions

This assignment demonstrates automated hyperparameter optimization using **Optuna** TPE (Tree-structured Parzen Estimator) with **ConvNeXt** models, integrated into a **GitHub Actions** CI/CD pipeline.

## 📋 Assignment Requirements

| Requirement | Status |
|-------------|--------|
| ✅ Run Hyper Param Optimization with GitHub Actions | Implemented |
| ✅ CI/CD Action adds comment with hparams & accuracy table | Implemented |
| ✅ Plot combined metrics (val/loss & val/acc) | Implemented |
| ✅ Optimize params for ConvNeXt | Implemented |
| ✅ Use DVC for data | Implemented |
| ✅ At least 10 experiments | 10 trials configured |
| ✅ Each experiment runs 2 epochs | Configured |

## 🏗️ Model: ConvNeXt

Using **ConvNeXt** variants from timm (small and efficient):
- `convnext_atto` - Smallest (5.7M params)
- `convnext_femto` - Tiny (7.1M params)
- `convnext_pico` - Small (9.1M params)

Reference: [timm/convnext.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py)

## 🔍 Hyperparameter Search Space

```yaml
params:
  model.lr: tag(log, interval(1e-5, 1e-2))
  model.weight_decay: tag(log, interval(1e-6, 1e-3))
  data.batch_size: choice(16, 32, 64)
  model.patience: choice(3, 5, 10)
  model.base_model: choice(convnext_atto, convnext_femto, convnext_pico)
```

## 🚀 Local Testing

```bash
# Install dependencies
uv sync

# Pull data
dvc pull

# Run single experiment
uv run python src/train.py experiment=catdog_ex model.base_model=convnext_atto trainer.max_epochs=2

# Run Optuna hyperparameter search (10 trials)
uv run python src/train.py -m hparams_search=convnext_optuna

# Generate report
uv run python scripts/generate_hparam_report.py
```

## 📊 GitHub Actions Workflow

The workflow (`.github/workflows/s07-hparam-optuna.yml`):

1. **Pulls data** from DVC (Google Drive)
2. **Runs 10 Optuna trials** with TPE sampler
3. **Generates report** with:
   - Hyperparameter table with test accuracy
   - Combined val/loss and val/acc plots
   - Best configuration
4. **Posts comment** on PR/commit with results

## 📁 Project Structure

```
z_assingment/
├── configs/
│   ├── hparams_search/
│   │   └── convnext_optuna.yaml    # Optuna config
│   ├── experiment/
│   ├── model/
│   └── ...
├── scripts/
│   └── generate_hparam_report.py   # Report generator
├── src/
│   ├── train.py                    # Training script (returns metric for Optuna)
│   ├── models/
│   └── datamodules/
├── .gitignore
├── pyproject.toml
└── README.md
```

## 📈 Expected Output

The GitHub Actions workflow posts a comment like:

```markdown
# 🎯 S07 ConvNeXt Hyperparameter Optimization Results

## 📊 Results Table

| Run | Test Acc | lr | weight_decay | batch_size | base_model |
|-----|----------|----|--------------|-----------:|------------|
| 3   | 0.9850   | 0.0003 | 0.00001 | 32 | convnext_pico |
| 1   | 0.9750   | 0.001  | 0.0001  | 64 | convnext_atto |
| ... | ...      | ...    | ...     | ...| ...        |

## 🏆 Best Configuration
- lr: 0.0003
- weight_decay: 0.00001
- batch_size: 32
- base_model: convnext_pico
```

## 🔧 Dependencies

```toml
dependencies = [
    "hydra-optuna-sweeper>=1.2.0",
    "lightning[extra]>=2.5.1",
    "timm>=1.0.15",
    "dvc>=3.65.0",
    "dvc-gdrive>=3.0.1",
]
```

## 📝 Notes

- **Optuna TPE**: Uses Bayesian optimization to find good hyperparameters faster than grid search
- **10 trials**: 3 random warmup + 7 optimized trials
- **ConvNeXt**: Modern CNN architecture, efficient and accurate
- **DVC**: Data versioned with Google Drive storage
