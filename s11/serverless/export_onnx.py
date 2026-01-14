import torch
import os

def export_model_to_onnx(
    traced_model_path="./model.pt", 
    output_path="./model.onnx"
):
    print(f"Loading traced model from {traced_model_path}...")
    try:
        model = torch.jit.load(traced_model_path)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create dummy input tensor matching your model's expected input size (1, 3, 160, 160)
    # The previous code used resize (160, 160)
    dummy_input = torch.randn(1, 3, 160, 160)

    print(f"Exporting to ONNX at {output_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            verbose=False,
            input_names=['input'],
            output_names=['output'],
        )
        print(f"Model exported successfully to {output_path}")
    except Exception as e:
        print(f"Error during export: {e}")

if __name__ == "__main__":
    export_model_to_onnx()
