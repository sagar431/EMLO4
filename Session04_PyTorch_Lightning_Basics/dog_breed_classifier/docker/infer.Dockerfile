FROM dog-breed-base:latest

# Mount points for input images, checkpoints, and predictions
VOLUME ["/app/input", "/app/checkpoints", "/app/predictions"]

# Command to run inference
CMD ["python", "src/infer.py", "-input_folder", "/app/input", "-output_folder", "/app/predictions", "-ckpt", "/app/checkpoints/best_model.ckpt"]
