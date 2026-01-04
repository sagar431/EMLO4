#!/usr/bin/env python3
"""
Script to generate hyperparameter optimization report.
Creates a markdown table with all hparams and test accuracy,
plus combined plots for val/loss and val/acc across all runs.
"""

import os
import json
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt



def find_multirun_dirs(base_dir: str = "logs/train/multiruns") -> list:
    """Find all multirun experiment directories."""
    pattern = os.path.join(base_dir, "*", "*")
    return sorted(glob.glob(pattern))


def load_metrics_from_csv(run_dir: str) -> pd.DataFrame:
    """Load metrics from CSV logger output."""
    csv_dir = os.path.join(run_dir, "csv")
    metrics_file = os.path.join(csv_dir, "version_0", "metrics.csv")
    
    if os.path.exists(metrics_file):
        return pd.read_csv(metrics_file)
    return pd.DataFrame()


def load_hparams_from_hydra(run_dir: str) -> dict:
    """Load hyperparameters from Hydra config."""
    config_file = os.path.join(run_dir, ".hydra", "config.yaml")
    overrides_file = os.path.join(run_dir, ".hydra", "overrides.yaml")
    
    hparams = {}
    
    if os.path.exists(overrides_file):
        import yaml
        with open(overrides_file, 'r') as f:
            overrides = yaml.safe_load(f)
            if overrides:
                for override in overrides:
                    if '=' in override:
                        key, value = override.split('=', 1)
                        hparams[key] = value
    
    return hparams


def get_test_accuracy(run_dir: str) -> float:
    """Get test accuracy from run."""
    metrics_df = load_metrics_from_csv(run_dir)
    
    if not metrics_df.empty and 'test/acc' in metrics_df.columns:
        return metrics_df['test/acc'].dropna().iloc[-1] if len(metrics_df['test/acc'].dropna()) > 0 else None
    
    # Try from val/acc as fallback
    if not metrics_df.empty and 'val/acc' in metrics_df.columns:
        return metrics_df['val/acc'].dropna().iloc[-1] if len(metrics_df['val/acc'].dropna()) > 0 else None
    
    return None


def plot_combined_metrics(run_dirs: list, output_dir: str = "outputs"):
    """Plot combined val/loss and val/acc for all runs."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(range(len(run_dirs)))
    
    for i, run_dir in enumerate(run_dirs):
        run_name = f"Run {i+1}"
        metrics_df = load_metrics_from_csv(run_dir)
        
        if metrics_df.empty:
            continue
        
        # Plot val/loss
        if 'val/loss' in metrics_df.columns:
            val_loss = metrics_df[metrics_df['val/loss'].notna()]
            if not val_loss.empty:
                axes[0].plot(
                    range(len(val_loss)), 
                    val_loss['val/loss'].values, 
                    label=run_name, 
                    color=colors[i],
                    marker='o',
                    markersize=4
                )
        
        # Plot val/acc
        if 'val/acc' in metrics_df.columns:
            val_acc = metrics_df[metrics_df['val/acc'].notna()]
            if not val_acc.empty:
                axes[1].plot(
                    range(len(val_acc)), 
                    val_acc['val/acc'].values, 
                    label=run_name, 
                    color=colors[i],
                    marker='o',
                    markersize=4
                )
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Validation Loss Across All Runs')
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Validation Accuracy Across All Runs')
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined metrics plot to {output_dir}/combined_metrics.png")


def generate_report(run_dirs: list, output_file: str = "outputs/hparam_report.md"):
    """Generate markdown report with hyperparameters and results."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    results = []
    
    for i, run_dir in enumerate(run_dirs):
        hparams = load_hparams_from_hydra(run_dir)
        test_acc = get_test_accuracy(run_dir)
        
        result = {
            'Run': i + 1,
            'Test Acc': f"{test_acc:.4f}" if test_acc else "N/A",
            **{k.replace('model.', '').replace('data.', ''): v for k, v in hparams.items() 
               if k.startswith('model.') or k.startswith('data.')}
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Sort by test accuracy
    if 'Test Acc' in df.columns:
        df_sorted = df.sort_values('Test Acc', ascending=False, key=lambda x: pd.to_numeric(x.replace('N/A', '-1')))
    else:
        df_sorted = df
    
    # Generate markdown
    with open(output_file, 'w') as f:
        f.write("# 🎯 Hyperparameter Optimization Results\n\n")
        f.write("## 📊 Results Table\n\n")
        f.write(df_sorted.to_markdown(index=False))
        f.write("\n\n")
        f.write("## 📈 Combined Metrics\n\n")
        f.write("### Validation Loss & Accuracy Across All Runs\n\n")
        f.write("![Combined Metrics](combined_metrics.png)\n\n")
        f.write("## 🏆 Best Configuration\n\n")
        if len(df_sorted) > 0:
            best = df_sorted.iloc[0]
            f.write(f"**Best Test Accuracy: {best['Test Acc']}**\n\n")
            f.write("```yaml\n")
            for col in df_sorted.columns:
                if col not in ['Run', 'Test Acc']:
                    f.write(f"{col}: {best[col]}\n")
            f.write("```\n")
    
    print(f"Saved report to {output_file}")
    return df_sorted


def main():
    """Main function to generate report."""
    print("🔍 Finding multirun directories...")
    run_dirs = find_multirun_dirs()
    
    if not run_dirs:
        print("No multirun directories found. Looking for individual runs...")
        run_dirs = find_multirun_dirs("logs/train/runs")
    
    if not run_dirs:
        print("❌ No runs found!")
        return
    
    print(f"✅ Found {len(run_dirs)} runs")
    
    print("\n📊 Generating combined metrics plot...")
    plot_combined_metrics(run_dirs)
    
    print("\n📝 Generating hyperparameter report...")
    df = generate_report(run_dirs)
    
    print("\n✅ Report generation complete!")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
