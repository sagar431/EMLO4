#!/usr/bin/env python3
"""Generate test metrics table in markdown format."""

import pandas as pd
import sys
import os


def main(metrics_file):
    """Extract test metrics and print as markdown table."""
    df = pd.read_csv(metrics_file)
    
    # Get test metrics (they are usually in the last rows)
    test_data = df[df['test/acc'].notna()].copy()
    
    if test_data.empty:
        print("| Metric | Value |")
        print("|--------|-------|")
        print("| Test Accuracy | N/A (test not run) |")
        print("| Test Loss | N/A (test not run) |")
        return
    
    # Get the final test metrics
    final_test = test_data.iloc[-1]
    test_acc = final_test['test/acc']
    test_loss = final_test['test/loss']
    
    # Format as percentage for accuracy
    test_acc_pct = f"{test_acc * 100:.2f}%"
    test_loss_fmt = f"{test_loss:.4f}"
    
    # Print markdown table
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| 🎯 **Test Accuracy** | **{test_acc_pct}** |")
    print(f"| 📉 **Test Loss** | **{test_loss_fmt}** |")
    
    # Also print training summary
    train_data = df[df['train/acc'].notna()].copy()
    if not train_data.empty:
        final_train_acc = train_data['train/acc'].iloc[-1]
        final_train_loss = train_data['train/loss'].iloc[-1]
        print(f"| 📈 Final Train Accuracy | {final_train_acc * 100:.2f}% |")
        print(f"| 📉 Final Train Loss | {final_train_loss:.4f} |")
    
    # Validation metrics
    val_data = df[df['val/acc'].notna()].copy()
    if not val_data.empty:
        final_val_acc = val_data['val/acc'].iloc[-1]
        final_val_loss = val_data['val/loss'].iloc[-1]
        print(f"| ✅ Final Val Accuracy | {final_val_acc * 100:.2f}% |")
        print(f"| 📉 Final Val Loss | {final_val_loss:.4f} |")
    
    # Number of epochs/steps
    max_epoch = df['epoch'].max()
    max_step = df['step'].max()
    print(f"| 🔄 Total Epochs | {int(max_epoch) + 1} |")
    print(f"| 👣 Total Steps | {int(max_step)} |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("| Metric | Value |")
        print("|--------|-------|")
        print("| Error | No metrics file provided |")
        sys.exit(1)

    metrics_file = sys.argv[1]
    if not os.path.exists(metrics_file):
        print("| Metric | Value |")
        print("|--------|-------|")
        print(f"| Error | File {metrics_file} not found |")
        sys.exit(1)

    main(metrics_file)
