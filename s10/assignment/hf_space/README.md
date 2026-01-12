---
title: Cat Dog Classifier
emoji: 🐱🐶
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🐱🐶 Cat vs Dog Classifier

A deep learning model that classifies images as either cats or dogs.

## Features

- **Fast Inference**: Uses TorchScript traced model for optimized performance
- **Easy to Use**: Simple upload interface powered by Gradio
- **Accurate**: Trained on a large cat-dog dataset

## How to Use

1. Upload an image of a cat or dog
2. Click "Submit" or wait for auto-processing
3. View the prediction with confidence scores

## Model Details

- **Architecture**: ResNet-18
- **Input Size**: 224x224 pixels
- **Output**: Binary classification (Cat/Dog)
- **Framework**: PyTorch + TorchScript

## Training

This model was trained as part of EMLO4 Session 10 Assignment on CI/CD deployment.

## License

MIT License
