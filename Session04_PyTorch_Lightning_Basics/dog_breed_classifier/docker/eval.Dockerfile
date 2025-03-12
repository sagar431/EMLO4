FROM dog-breed-base:latest

# Mount points for data and checkpoints
VOLUME ["/app/data", "/app/checkpoints", "/app/predictions"]

# Command to run evaluation
CMD ["python", "src/eval.py"]
