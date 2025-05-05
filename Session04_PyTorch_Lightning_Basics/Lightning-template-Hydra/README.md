# Cat/Dog Classifier

A PyTorch Lightning implementation of a cat/dog image classifier using transfer learning with ResNet18.

## Project Structure
```
├── pyproject.toml        # Project dependencies
├── README.md            # This file
├── src
│   ├── datamodules/    # Lightning DataModules
│   ├── models/         # Lightning Modules
│   ├── utils/          # Utility functions
│   ├── train.py        # Training script
│   └── infer.py        # Inference script
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -e .
```

## Training

To train the model:

```bash
python src/train.py
```

The model will be trained using the cats and dogs dataset. Training logs and checkpoints will be saved in the `logs` directory.

## Inference

To run inference on new images:

```bash
python src/infer.py --input_folder samples --output_folder predictions --ckpt_path "/path/to/checkpoint.ckpt"
```

Arguments:
- `--input_folder`: Directory containing images to classify
- `--output_folder`: Directory where predictions will be saved
- `--ckpt_path`: Path to the trained model checkpoint

## Features

- Transfer learning using ResNet18
- Rich progress bars and logging
- TensorBoard integration
- Model checkpointing
- Test set evaluation
- Inference with confidence scores
