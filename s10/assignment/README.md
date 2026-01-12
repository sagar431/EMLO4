# 🐱🐶 Cat-Dog Classifier - CI/CD Deployment Assignment

EMLO4 Session 10: Deployment with Gradio

## 📋 Assignment Overview

This assignment implements a complete CI/CD pipeline for deploying a Cat-Dog classifier as a Gradio app to HuggingFace Spaces.

### Tasks Completed:
- ✅ Python script to trace/script the trained model
- ✅ Gradio app to serve the classifier
- ✅ HuggingFace Spaces deployment configuration
- ✅ GitHub Actions workflow for CI/CD

## 📁 Project Structure

```
assignment/
├── README.md                    # This file
├── src/
│   └── trace_model.py           # Script to trace/script model
├── hf_space/
│   ├── app.py                   # Gradio app for HuggingFace Spaces
│   ├── requirements.txt         # Python dependencies
│   ├── README.md                # HuggingFace Space README (metadata)
│   └── model.pt                 # Traced model (generated)
├── examples/
│   ├── cat.jpg                  # Example cat image
│   └── dog.jpg                  # Example dog image
└── .github/
    └── workflows/
        └── deploy.yml           # GitHub Actions CI/CD workflow
```

## 🚀 Quick Start

### 1. Trace Your Trained Model

```bash
# From the training project directory
python src/trace_model.py \
    --ckpt_path logs/train/runs/YYYY-MM-DD/checkpoints/last.ckpt \
    --output_path hf_space/model.pt \
    --model_name resnet18 \
    --num_classes 2 \
    --input_size 224
```

### 2. Test Locally

```bash
cd hf_space
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860 in your browser.

### 3. Deploy to HuggingFace Spaces

**Manual Deployment:**
```bash
pip install huggingface_hub
huggingface-cli login
cd hf_space
huggingface-cli upload YOUR_USERNAME/cat-dog-classifier . --repo-type space
```

**Automatic Deployment (GitHub Actions):**
1. Add secrets to your GitHub repository:
   - `HF_TOKEN`: Your HuggingFace access token
   - `HF_USERNAME`: Your HuggingFace username
2. Push to main branch
3. GitHub Actions will automatically deploy

## 🔧 Configuration

### Model Tracing Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--ckpt_path` | Path to checkpoint file | Required |
| `--output_path` | Output path for traced model | `model.pt` |
| `--model_name` | timm model name | `resnet18` |
| `--num_classes` | Number of output classes | `2` |
| `--input_size` | Input image size | `224` |
| `--method` | Tracing method (`trace` or `script`) | `trace` |

### GitHub Actions Secrets

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | HuggingFace access token (write permission) |
| `HF_USERNAME` | Your HuggingFace username |

## 📊 CI/CD Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Test     │ --> │   Train*    │ --> │    Trace    │ --> │   Deploy    │
│  (pytest)   │     │ (optional)  │     │ (TorchScript)│     │ (HF Spaces) │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘

* Training job runs only on manual workflow dispatch
```

### Pipeline Jobs:

1. **Test**: Runs pytest tests
2. **Train**: Trains the model (manual trigger only)
3. **Trace**: Converts model to TorchScript
4. **Deploy**: Uploads to HuggingFace Spaces

## 🐱 TorchScript Explanation

TorchScript converts PyTorch models to a serializable format that can run without Python.

### Tracing vs Scripting

| Method | How it works | Best for |
|--------|--------------|----------|
| **Trace** | Records operations during forward pass | Simple models, no dynamic control flow |
| **Script** | Parses Python source code | Models with if/else based on data |

### Example:

```python
import torch

# Load model and set to eval mode
model = YourModel()
model.eval()

# Create example input
example_input = torch.randn(1, 3, 224, 224)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Save for deployment
torch.jit.save(traced_model, "model.pt")

# Load and use (no Python code needed!)
loaded_model = torch.jit.load("model.pt")
output = loaded_model(input_tensor)
```

## 🌐 Deployment

### HuggingFace Spaces

Your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/cat-dog-classifier
```

### Features:
- Free GPU inference (Zero GPU)
- Automatic scaling
- Public URL for sharing
- Version control

## 📝 Notes

- Make sure your HuggingFace token has write permissions
- The traced model file (`model.pt`) can be large (~100MB for ResNet)
- Consider using Git LFS for large files on HuggingFace

## 🔗 References

- [Gradio Documentation](https://www.gradio.app/)
- [HuggingFace Spaces](https://huggingface.co/docs/hub/spaces)
- [TorchScript Tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
- [GitHub Actions](https://docs.github.com/en/actions)

## 📄 License

MIT License

---

Made with ❤️ for EMLO4 Session 10 Assignment
