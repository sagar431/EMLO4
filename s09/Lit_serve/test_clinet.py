import requests
from urllib.request import urlopen
import base64

def test_single_image():
    # Get test image
    url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    img_data = urlopen(url).read()
    
    # Convert to base64 string
    img_bytes = base64.b64encode(img_data).decode('utf-8')
    
    # Send request
    response = requests.post(
        "http://localhost:8000/predict",
        json={"image": img_bytes}  # Send as JSON instead of files
    )
    
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        print("\nTop 5 Predictions:")
        for pred in predictions:
            print(f"{pred['label']}: {pred['probability']:.2%}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_single_image()