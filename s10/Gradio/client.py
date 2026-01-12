from gradio_client import Client, handle_file
import concurrent.futures
import time

def make_prediction(client, image_path, request_id):
    """Make a single prediction"""
    try:
        start = time.time()
        result = client.predict(
            image=handle_file(image_path),
            api_name="/predict"
        )
        elapsed = time.time() - start
        print(f"Request {request_id} completed in {elapsed:.2f}s")
        return result
    except Exception as e:
        return f"Error in request {request_id}: {str(e)}"

def main():
    # Use local file path instead of localhost URL
    image_path = '/home/ubuntu/EMLO4/s10/Gradio/golden_retriever.png'
    
    # Initialize client
    client = Client("http://127.0.0.1:7860/")
    
    print("\nSending 16 concurrent requests...")
    start_time = time.time()
    
    # Use ThreadPoolExecutor to send 16 requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(make_prediction, client, image_path, i+1) 
            for i in range(16)
        ]
        
        # Collect results as they complete
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error: {str(e)}")
    
    end_time = time.time()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"All 16 predictions completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per request: {(end_time - start_time) / 16:.2f} seconds")
    print(f"{'='*50}")
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"\nRequest {i+1}:")
        print(result)

if __name__ == "__main__":
    main()