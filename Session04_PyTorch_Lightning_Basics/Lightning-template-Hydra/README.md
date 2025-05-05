# Cat/Dog Image Classifier with PyTorch Lightning

A robust and efficient implementation of a cat/dog image classifier using PyTorch Lightning and transfer learning with ResNet18. This project demonstrates best practices in deep learning code organization, training, and inference.

## Training Results

![Training Progress](images/traning.png)

## Inference Examples

![Inference Results](images/inference.png)

Example predictions on test images:

| Cat Prediction | Dog Prediction |
|:-------------:|:-------------:|
| ![Cat Prediction](predictions/cat_prediction.png) | ![Dog Prediction](predictions/dog_prediction.png) |

## Project Structure
```
├── pyproject.toml        # Project dependencies
├── README.md            # Project documentation
├── src/
│   ├── datamodules/    # Lightning DataModules for data handling
│   ├── models/         # Model architectures and Lightning Modules
│   ├── utils/          # Utility functions and helpers
│   ├── train.py        # Training script
│   └── infer.py        # Inference script for predictions
├── data/               # Dataset directory
├── logs/               # Training logs and checkpoints
├── samples/            # Example images for inference
└── predictions/        # Model predictions output
```

## Features

- 🚀 **PyTorch Lightning** for clean, organized training code
- 🔄 **Transfer Learning** using ResNet18 architecture
- 📊 **Rich Progress Bars** and detailed logging
- 📈 **TensorBoard Integration** for monitoring training
- 💾 **Model Checkpointing** for saving best models
- 🎯 **Inference Pipeline** with confidence scores
- 🖼️ **Visualization** of predictions with matplotlib
- 🔍 **Detailed Logging** of training and inference

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Lightning-template-Hydra
```

2. Create and activate a virtual environment:
```bash
# Linux/Mac
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Training the Model

1. Prepare your dataset in the `data` directory with the following structure:
```
data/
    cats_and_dogs_filtered/
        train/
            cats/
            dogs/
        validation/
            cats/
            dogs/
```

2. Start training:
```bash
python src/train.py
```

The training script will:
- Use transfer learning with ResNet18
- Save checkpoints in the `logs` directory
- Log metrics and progress
- Validate on the validation set

## Running Inference

1. Place your test images in the `samples` directory

2. Run inference:
```bash
python src/infer.py \
    --input_folder samples \
    --output_folder predictions \
    --ckpt_path logs/catdog_classification/epoch=epoch=3-step=step=200-v1.ckpt
```

Arguments:
- `--input_folder`: Directory containing test images
- `--output_folder`: Directory for saving predictions
- `--ckpt_path`: Path to the trained model checkpoint

The inference script will:
- Load the trained model
- Process each image
- Generate visualizations with predictions
- Save results in the output folder

## Model Performance

The model achieves:
- High accuracy on the validation set
- Fast inference time
- Reliable confidence scores

## Customization

You can customize various aspects of training and inference:
- Model architecture in `src/models/`
- Data preprocessing in `src/datamodules/`
- Training parameters in `src/train.py`
- Inference visualization in `src/infer.py`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning
- torchvision
- matplotlib
- PIL

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent framework
- PyTorch Lightning team for the great library
- The deep learning community for valuable insights
