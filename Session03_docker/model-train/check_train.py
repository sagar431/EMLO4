import os
import sys

def main():
    model_dir = '/opt/mount/model'
    checkpoint_path = os.path.join(model_dir, 'mnist_cnn.pt')
    
    if os.path.exists(checkpoint_path):
        print("Checkpoint file found.")
        sys.exit(0)
    else:
        print("Checkpoint file not found!")
        sys.exit(1)

if __name__ == '__main__':
    main() 