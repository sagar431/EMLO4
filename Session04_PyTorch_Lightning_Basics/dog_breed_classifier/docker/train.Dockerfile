FROM dog-breed-base:latest

# Mount points for data and checkpoints
VOLUME ["/app/data", "/app/checkpoints"]

# Command to run training
CMD ["python", "src/train.py"]
