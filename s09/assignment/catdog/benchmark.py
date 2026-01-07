"""
Comprehensive Benchmark Script for Cat-Dog Classifier

Measures:
1. Baseline model throughput (theoretical maximum)
2. API server throughput (real-world performance)
3. CPU/GPU utilization
4. Generates comparison plots
"""

import base64
import concurrent.futures
import time
import numpy as np
import requests
import torch
import timm
import matplotlib.pyplot as plt
from urllib.request import urlopen
from PIL import Image
import psutil

try:
    import gpustat
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Constants
SERVER_URL = "http://localhost:8000/predict"
# Use local test image
TEST_IMAGE_PATH = "test_cat.png"


def get_baseline_throughput(batch_size, num_iterations=10):
    """Calculate baseline model throughput without API overhead"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model and move to device (same as server)
    model = timm.create_model('resnet18', pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Create random input data
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    throughputs = []
    
    # Warm-up run
    with torch.no_grad():
        model(x)
    
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        reqs_per_sec = batch_size / (t1 - t0)
        throughputs.append(reqs_per_sec)
    
    return np.mean(throughputs)


def prepare_test_payload():
    """Prepare a test image payload from local file"""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, TEST_IMAGE_PATH)
    with open(image_path, 'rb') as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode('utf-8')


def send_request(payload):
    """Send a single request and measure response time"""
    start_time = time.time()
    response = requests.post(SERVER_URL, json={"image": payload})
    end_time = time.time()
    return end_time - start_time, response.status_code


def get_system_metrics():
    """Get current GPU and CPU usage"""
    metrics = {"cpu_usage": psutil.cpu_percent(0.1)}
    if GPU_AVAILABLE:
        try:
            gpu_stats = gpustat.GPUStatCollection.new_query()
            metrics["gpu_usage"] = sum([gpu.utilization for gpu in gpu_stats.gpus])
        except Exception:
            metrics["gpu_usage"] = -1
    else:
        metrics["gpu_usage"] = -1
    return metrics


def benchmark_api(num_requests=100, concurrency_level=10):
    """Benchmark the API server"""
    payload = prepare_test_payload()
    system_metrics = []
    
    start_benchmark_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(send_request, payload) for _ in range(num_requests)]
        response_times = []
        status_codes = []
        
        while any(not f.done() for f in futures):
            system_metrics.append(get_system_metrics())
            time.sleep(0.1)
        
        for future in futures:
            response_time, status_code = future.result()
            response_times.append(response_time)
            status_codes.append(status_code)
    
    end_benchmark_time = time.time()
    total_benchmark_time = end_benchmark_time - start_benchmark_time
    
    avg_cpu = np.mean([m["cpu_usage"] for m in system_metrics]) if system_metrics else 0
    avg_gpu = np.mean([m["gpu_usage"] for m in system_metrics]) if GPU_AVAILABLE and system_metrics else -1
    
    return {
        "total_requests": num_requests,
        "concurrency_level": concurrency_level,
        "total_time": total_benchmark_time,
        "avg_response_time": np.mean(response_times) * 1000,
        "p95_response_time": np.percentile(response_times, 95) * 1000,
        "p99_response_time": np.percentile(response_times, 99) * 1000,
        "success_rate": (status_codes.count(200) / num_requests) * 100,
        "requests_per_second": num_requests / total_benchmark_time,
        "avg_cpu_usage": avg_cpu,
        "avg_gpu_usage": avg_gpu
    }


def run_benchmarks():
    """Run comprehensive benchmarks and create plots"""
    print("=" * 60)
    print("Cat-Dog Classifier Benchmark")
    print("=" * 60)
    
    # Test different batch sizes for baseline throughput
    batch_sizes = [1, 8, 32, 64, 128]
    baseline_throughput = []
    
    print("\n📊 Running baseline throughput tests...")
    for batch_size in batch_sizes:
        reqs_per_sec = get_baseline_throughput(batch_size)
        baseline_throughput.append(reqs_per_sec)
        print(f"   Batch size {batch_size:3d}: {reqs_per_sec:8.2f} reqs/sec")
    
    theoretical_max = max(baseline_throughput)
    print(f"\n   🎯 Theoretical Maximum: {theoretical_max:.2f} reqs/sec")
    
    # Test different concurrency levels for API
    concurrency_levels = [1, 8, 32, 64]
    api_throughput = []
    cpu_usage = []
    gpu_usage = []
    response_times = []
    
    print("\n🚀 Running API benchmarks...")
    for concurrency in concurrency_levels:
        metrics = benchmark_api(num_requests=128, concurrency_level=concurrency)
        api_throughput.append(metrics["requests_per_second"])
        cpu_usage.append(metrics["avg_cpu_usage"])
        gpu_usage.append(metrics["avg_gpu_usage"])
        response_times.append(metrics["avg_response_time"])
        
        efficiency = (metrics["requests_per_second"] / theoretical_max) * 100
        print(f"   Concurrency {concurrency:2d}: {metrics['requests_per_second']:6.2f} reqs/sec "
              f"(Efficiency: {efficiency:5.1f}%) | "
              f"CPU: {metrics['avg_cpu_usage']:5.1f}% | GPU: {metrics['avg_gpu_usage']:5.1f}% | "
              f"p95: {metrics['p95_response_time']:6.1f}ms")
    
    best_api = max(api_throughput)
    best_efficiency = (best_api / theoretical_max) * 100
    
    print(f"\n📈 Summary:")
    print(f"   Theoretical Max:  {theoretical_max:.2f} reqs/sec")
    print(f"   Best API:         {best_api:.2f} reqs/sec")
    print(f"   Efficiency:       {best_efficiency:.1f}%")
    print(f"   Performance Gap:  {100 - best_efficiency:.1f}% room for optimization")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Throughput comparison
    ax1 = axes[0, 0]
    ax1.plot(batch_sizes, baseline_throughput, 'b-o', label='Baseline Model (Theoretical Max)', linewidth=2, markersize=8)
    ax1.axhline(y=best_api, color='r', linestyle='--', label=f'Best API: {best_api:.1f} reqs/sec')
    ax1.fill_between(batch_sizes, baseline_throughput, alpha=0.3)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (requests/second)')
    ax1.set_title('Baseline Model Throughput vs Batch Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # API throughput by concurrency
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(concurrency_levels)), api_throughput, color='coral', edgecolor='black')
    ax2.axhline(y=theoretical_max, color='blue', linestyle='--', label=f'Theoretical Max: {theoretical_max:.1f}')
    ax2.set_xticks(range(len(concurrency_levels)))
    ax2.set_xticklabels([f'C={c}' for c in concurrency_levels])
    ax2.set_xlabel('Concurrency Level')
    ax2.set_ylabel('Throughput (requests/second)')
    ax2.set_title('API Server Throughput')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add efficiency labels on bars
    for bar, throughput in zip(bars, api_throughput):
        eff = (throughput / theoretical_max) * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{eff:.0f}%', ha='center', va='bottom', fontsize=10)
    
    # CPU/GPU Usage
    ax3 = axes[1, 0]
    x = np.arange(len(concurrency_levels))
    width = 0.35
    ax3.bar(x - width/2, cpu_usage, width, label='CPU Usage', color='green', alpha=0.7)
    ax3.bar(x + width/2, gpu_usage, width, label='GPU Usage', color='purple', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'C={c}' for c in concurrency_levels])
    ax3.set_xlabel('Concurrency Level')
    ax3.set_ylabel('Usage (%)')
    ax3.set_title('Resource Utilization')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Response times
    ax4 = axes[1, 1]
    ax4.plot(concurrency_levels, response_times, 'g-o', linewidth=2, markersize=8)
    ax4.set_xlabel('Concurrency Level')
    ax4.set_ylabel('Average Response Time (ms)')
    ax4.set_title('Response Time vs Concurrency')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Cat-Dog Classifier Performance Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Benchmark complete! Results saved to benchmark_results.png")
    print("=" * 60)
    
    # Return results for README generation
    return {
        "batch_sizes": batch_sizes,
        "baseline_throughput": baseline_throughput,
        "theoretical_max": theoretical_max,
        "concurrency_levels": concurrency_levels,
        "api_throughput": api_throughput,
        "cpu_usage": cpu_usage,
        "gpu_usage": gpu_usage,
        "best_api": best_api,
        "best_efficiency": best_efficiency
    }


if __name__ == "__main__":
    run_benchmarks()
