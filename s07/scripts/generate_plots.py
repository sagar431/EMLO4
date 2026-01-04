#!/usr/bin/env python3
"""Generate beautiful training plots from metrics.csv using matplotlib."""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


def create_training_accuracy_plot(df):
    """Create training accuracy plot."""
    # Filter rows with training accuracy
    train_data = df[df['train/acc'].notna()].copy()
    
    if train_data.empty:
        print("No training accuracy data found")
        return
    
    fig, ax = plt.subplots()
    
    ax.plot(train_data['step'], train_data['train/acc'], 
            color='#2ecc71', linewidth=2, marker='o', markersize=3,
            label='Training Accuracy')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy Over Time')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Add a subtle fill under the curve
    ax.fill_between(train_data['step'], train_data['train/acc'], alpha=0.2, color='#2ecc71')
    
    plt.tight_layout()
    plt.savefig('train_acc_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: train_acc_plot.png")


def create_training_loss_plot(df):
    """Create training loss plot."""
    train_data = df[df['train/loss'].notna()].copy()
    
    if train_data.empty:
        print("No training loss data found")
        return
    
    fig, ax = plt.subplots()
    
    ax.plot(train_data['step'], train_data['train/loss'], 
            color='#e74c3c', linewidth=2, marker='o', markersize=3,
            label='Training Loss')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add a subtle fill under the curve
    ax.fill_between(train_data['step'], train_data['train/loss'], alpha=0.2, color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig('train_loss_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: train_loss_plot.png")


def create_validation_accuracy_plot(df):
    """Create validation accuracy plot."""
    val_data = df[df['val/acc'].notna()].copy()
    
    if val_data.empty:
        print("No validation accuracy data found")
        return
    
    fig, ax = plt.subplots()
    
    ax.plot(val_data['step'], val_data['val/acc'], 
            color='#3498db', linewidth=2, marker='s', markersize=5,
            label='Validation Accuracy')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy Over Time')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    ax.fill_between(val_data['step'], val_data['val/acc'], alpha=0.2, color='#3498db')
    
    plt.tight_layout()
    plt.savefig('val_acc_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: val_acc_plot.png")


def create_validation_loss_plot(df):
    """Create validation loss plot."""
    val_data = df[df['val/loss'].notna()].copy()
    
    if val_data.empty:
        print("No validation loss data found")
        return
    
    fig, ax = plt.subplots()
    
    ax.plot(val_data['step'], val_data['val/loss'], 
            color='#9b59b6', linewidth=2, marker='s', markersize=5,
            label='Validation Loss')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss Over Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    ax.fill_between(val_data['step'], val_data['val/loss'], alpha=0.2, color='#9b59b6')
    
    plt.tight_layout()
    plt.savefig('val_loss_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: val_loss_plot.png")


def create_combined_accuracy_plot(df):
    """Create combined training and validation accuracy plot."""
    train_data = df[df['train/acc'].notna()].copy()
    val_data = df[df['val/acc'].notna()].copy()
    
    if train_data.empty and val_data.empty:
        print("No accuracy data found")
        return
    
    fig, ax = plt.subplots()
    
    if not train_data.empty:
        ax.plot(train_data['step'], train_data['train/acc'], 
                color='#2ecc71', linewidth=2, marker='o', markersize=3,
                label='Training Accuracy', alpha=0.8)
    
    if not val_data.empty:
        ax.plot(val_data['step'], val_data['val/acc'], 
                color='#3498db', linewidth=2, marker='s', markersize=5,
                label='Validation Accuracy', alpha=0.8)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training vs Validation Accuracy')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('combined_acc_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: combined_acc_plot.png")


def create_combined_loss_plot(df):
    """Create combined training and validation loss plot."""
    train_data = df[df['train/loss'].notna()].copy()
    val_data = df[df['val/loss'].notna()].copy()
    
    if train_data.empty and val_data.empty:
        print("No loss data found")
        return
    
    fig, ax = plt.subplots()
    
    if not train_data.empty:
        ax.plot(train_data['step'], train_data['train/loss'], 
                color='#e74c3c', linewidth=2, marker='o', markersize=3,
                label='Training Loss', alpha=0.8)
    
    if not val_data.empty:
        ax.plot(val_data['step'], val_data['val/loss'], 
                color='#9b59b6', linewidth=2, marker='s', markersize=5,
                label='Validation Loss', alpha=0.8)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('combined_loss_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: combined_loss_plot.png")


def main(metrics_file):
    """Main function to generate all plots."""
    print(f"Reading metrics from: {metrics_file}")
    
    # Read the CSV file
    df = pd.read_csv(metrics_file)
    print(f"Loaded {len(df)} rows of metrics data")
    print(f"Columns: {list(df.columns)}")
    
    # Generate individual plots
    create_training_accuracy_plot(df)
    create_training_loss_plot(df)
    create_validation_accuracy_plot(df)
    create_validation_loss_plot(df)
    
    # Generate combined plots
    create_combined_accuracy_plot(df)
    create_combined_loss_plot(df)
    
    print("\n✅ All plots generated successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_plots.py <path_to_metrics_csv>")
        sys.exit(1)

    metrics_file = sys.argv[1]
    if not os.path.exists(metrics_file):
        print(f"Error: File {metrics_file} does not exist.")
        sys.exit(1)

    main(metrics_file)
