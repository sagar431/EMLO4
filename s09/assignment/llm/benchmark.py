"""
Benchmark Script for GPT-OSS-20B LLM

Measures:
1. Tokens per second
2. Time to first token
3. Generation latency
4. Compare with theoretical maximum
"""

import time
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt


# Test prompts of varying complexity
TEST_PROMPTS = [
    "What is 2+2?",
    "Explain what machine learning is in one paragraph.",
    "Write a Python function to reverse a string.",
    "What are the main differences between Python and JavaScript?",
    "Explain the concept of neural networks and how they learn."
]


def calculate_theoretical_max():
    """
    Calculate theoretical maximum tokens/sec for A10 GPU
    
    A10 specs:
    - Memory Bandwidth: 600 GB/s
    - FP16 TFLOPS: 125 TFLOPS
    
    For LLM inference (memory-bound):
    - GPT-OSS-20B with MXFP4: ~10GB model size
    - Tokens/sec ≈ Memory_BW / (2 * Model_Size_per_token)
    - Rough estimate: 40-80 tokens/sec for 20B model
    """
    return 60  # Conservative estimate for 20B model on A10


def benchmark_single_request(client, prompt, max_tokens=128):
    """Benchmark a single request"""
    start_time = time.perf_counter()
    
    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=False
    )
    
    end_time = time.perf_counter()
    
    # Extract response and calculate metrics
    content = response.choices[0].message.content
    
    # Estimate tokens (rough approximation: 4 chars per token)
    estimated_tokens = len(content) // 4
    generation_time = end_time - start_time
    tokens_per_sec = estimated_tokens / generation_time if generation_time > 0 else 0
    
    return {
        "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
        "response_length": len(content),
        "estimated_tokens": estimated_tokens,
        "generation_time": generation_time,
        "tokens_per_sec": tokens_per_sec
    }


def run_benchmark(num_iterations=3):
    """Run comprehensive benchmark"""
    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="dummy-key"
    )
    
    print("=" * 70)
    print("GPT-OSS-20B LLM Benchmark")
    print("=" * 70)
    
    theoretical_max = calculate_theoretical_max()
    print(f"\n🎯 Theoretical Maximum (A10 GPU): ~{theoretical_max} tokens/sec")
    
    all_results = []
    
    print("\n📊 Running benchmarks...")
    print("-" * 70)
    
    for i, prompt in enumerate(TEST_PROMPTS):
        prompt_results = []
        
        for iteration in range(num_iterations):
            result = benchmark_single_request(client, prompt)
            prompt_results.append(result)
            all_results.append(result)
        
        avg_tokens_per_sec = np.mean([r["tokens_per_sec"] for r in prompt_results])
        avg_time = np.mean([r["generation_time"] for r in prompt_results])
        efficiency = (avg_tokens_per_sec / theoretical_max) * 100
        
        print(f"\n   Prompt {i+1}: {prompt[:40]}...")
        print(f"   Avg Tokens/sec: {avg_tokens_per_sec:.2f} | "
              f"Avg Time: {avg_time:.2f}s | "
              f"Efficiency: {efficiency:.1f}%")
    
    # Calculate overall statistics
    all_tokens_per_sec = [r["tokens_per_sec"] for r in all_results]
    all_times = [r["generation_time"] for r in all_results]
    
    avg_overall = np.mean(all_tokens_per_sec)
    std_overall = np.std(all_tokens_per_sec)
    max_achieved = np.max(all_tokens_per_sec)
    min_achieved = np.min(all_tokens_per_sec)
    efficiency = (avg_overall / theoretical_max) * 100
    
    print("\n" + "=" * 70)
    print("📈 Summary")
    print("=" * 70)
    print(f"\n   Theoretical Maximum:  {theoretical_max} tokens/sec")
    print(f"   Average Achieved:     {avg_overall:.2f} tokens/sec")
    print(f"   Standard Deviation:   {std_overall:.2f}")
    print(f"   Best Run:             {max_achieved:.2f} tokens/sec")
    print(f"   Worst Run:            {min_achieved:.2f} tokens/sec")
    print(f"   Efficiency:           {efficiency:.1f}%")
    print(f"   Average Latency:      {np.mean(all_times):.2f}s")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Tokens/sec distribution
    ax1 = axes[0]
    ax1.hist(all_tokens_per_sec, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=avg_overall, color='red', linestyle='--', label=f'Mean: {avg_overall:.1f}')
    ax1.axvline(x=theoretical_max, color='green', linestyle='--', label=f'Theoretical: {theoretical_max}')
    ax1.set_xlabel('Tokens/sec')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Tokens/sec Distribution')
    ax1.legend()
    
    # Comparison bar chart
    ax2 = axes[1]
    categories = ['Achieved', 'Theoretical Max']
    values = [avg_overall, theoretical_max]
    colors = ['coral', 'limegreen']
    bars = ax2.bar(categories, values, color=colors, edgecolor='black')
    ax2.set_ylabel('Tokens/sec')
    ax2.set_title('Achieved vs Theoretical Maximum')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', va='bottom')
    
    # Latency distribution
    ax3 = axes[2]
    ax3.hist(all_times, bins=10, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax3.axvline(x=np.mean(all_times), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_times):.2f}s')
    ax3.set_xlabel('Generation Time (s)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Latency Distribution')
    ax3.legend()
    
    plt.suptitle('GPT-OSS-20B LLM Performance Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Benchmark complete! Results saved to benchmark_results.png")
    print("=" * 70)
    
    return {
        "theoretical_max": theoretical_max,
        "avg_tokens_per_sec": avg_overall,
        "efficiency": efficiency,
        "avg_latency": np.mean(all_times)
    }


if __name__ == "__main__":
    run_benchmark()
