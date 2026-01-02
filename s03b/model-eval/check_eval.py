import os
import sys
import json

def main():
    model_dir = '/opt/mount/model'
    eval_results_path = os.path.join(model_dir, 'eval_results.json')
    
    if not os.path.exists(eval_results_path):
        print("Evaluation results file not found!")
        sys.exit(1)
    
    # Read and display the evaluation results
    try:
        with open(eval_results_path, 'r') as f:
            results = json.load(f)
        
        print("Evaluation results found:")
        print(f"Test loss: {results['Test loss']:.4f}")
        print(f"Accuracy: {results['Accuracy']:.2f}%")
        sys.exit(0)
    except Exception as e:
        print(f"Error reading evaluation results: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
