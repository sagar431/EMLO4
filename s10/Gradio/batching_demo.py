#!/usr/bin/env python3
"""
BATCHING DEMO: Understanding the difference between single vs batched processing

This script shows what happens inside the model when processing images
"""

import torch
import time

# Simulate what happens WITHOUT batching
def process_without_batching(images):
    """
    OLD WAY: Process each image separately
    """
    print("\n🔄 WITHOUT BATCHING:")
    print("   Each image processed individually...")

    results = []
    total_time = 0

    for i, img in enumerate(images):
        # Simulate preprocessing + model inference time
        start = time.time()
        time.sleep(0.1)  # Simulate model inference time
        elapsed = time.time() - start
        total_time += elapsed

        results.append(f"Result for image {i+1}")
        print(".2f")

    print(".2f")
    return results, total_time

# Simulate what happens WITH batching
def process_with_batching(images, batch_size=4):
    """
    NEW WAY: Process images in batches
    """
    print(f"\n🚀 WITH BATCHING (batch_size={batch_size}):")
    print(f"   Processing {len(images)} images in batches...")

    results = []
    total_time = 0

    # Process in batches
    for batch_start in range(0, len(images), batch_size):
        batch_end = min(batch_start + batch_size, len(images))
        current_batch = images[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1

        print(f"\n   Batch {batch_num}: Processing {len(current_batch)} images together")

        # Simulate batch preprocessing + single model call
        start = time.time()
        time.sleep(0.15)  # Simulate batch processing (slightly longer due to overhead)
        elapsed = time.time() - start
        total_time += elapsed

        # Get results for entire batch
        for i in range(len(current_batch)):
            img_idx = batch_start + i + 1
            results.append(f"Result for image {img_idx}")
            print(".2f")

    print(".2f")
    print(f"   Total batches: {(len(images) + batch_size - 1) // batch_size}")
    return results, total_time

def main():
    # Simulate 16 concurrent requests (like your client sends)
    images = [f"image_{i+1}" for i in range(16)]

    print("🎯 BATCHING DEMO: 16 Images Processing Comparison")
    print("=" * 60)
    print(f"Processing {len(images)} images...")

    # Test without batching
    results1, time_no_batch = process_without_batching(images)

    # Test with batching (batch_size=4)
    results2, time_with_batch = process_with_batching(images, batch_size=4)

    # Calculate time differences
    time_saved = time_no_batch - time_with_batch
    speedup_ratio = time_no_batch / time_with_batch if time_with_batch > 0 else float('inf')
    improvement_percent = ((time_no_batch - time_with_batch) / time_no_batch) * 100

    print("\n" + "=" * 60)
    print("⏱️  PERFORMANCE COMPARISON:")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".1f")
    print(".1f")

    print("\n📊 SUMMARY:")
    print("   Without batching: 16 separate model calls")
    print("   With batching:     4 batches × 1 model call each")
    print("   GPU efficiency:    Much better with batching!")
    print("   Why? GPU processes multiple images in parallel")

    print("\n💡 KEY INSIGHT:")
    print("   Batching groups requests and processes them together")
    print("   Your 16 concurrent requests → 4 efficient batches")
    print("   Same results, much faster processing!")

    print("\n🎯 BOTTOM LINE:")
    print(".1f")
    print("   Perfect for concurrent requests! 🚀")

if __name__ == "__main__":
    main()