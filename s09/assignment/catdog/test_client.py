"""
Test client for Cat-Dog Classifier
Uses local test images for reliable testing
"""

import requests
import base64
import os


def test_image(image_path, expected_type):
    """Test with a local image file"""
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return
    
    # Read and encode local image
    with open(image_path, 'rb') as f:
        img_data = f.read()
    img_bytes = base64.b64encode(img_data).decode('utf-8')
    
    response = requests.post(
        "http://localhost:8000/predict",
        json={"image": img_bytes}
    )
    
    if response.status_code == 200:
        result = response.json()
        emoji = "🐱" if expected_type == "cat" else "🐕"
        status = "✅" if result['prediction'] == expected_type else "❌"
        
        print(f"\n{emoji} {expected_type.upper()} Image Test: {status}")
        print(f"   Prediction: {result['prediction'].upper()}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Cat: {result['cat_probability']:.2%} | Dog: {result['dog_probability']:.2%}")
        print(f"   Top ImageNet: {result['details']['top_imagenet_label']} ({result['details']['top_imagenet_confidence']:.2%})")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    print("=" * 50)
    print("Cat-Dog Classifier Test")
    print("=" * 50)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test cat image
    cat_path = os.path.join(script_dir, "test_cat.png")
    test_image(cat_path, "cat")
    
    # Test dog image
    dog_path = os.path.join(script_dir, "test_dog.png")
    test_image(dog_path, "dog")
    
    print("\n" + "=" * 50)
