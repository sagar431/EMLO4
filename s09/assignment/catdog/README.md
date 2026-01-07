# 🐱🐕 Cat-Dog Classifier with LitServe

A high-performance Cat-Dog image classifier deployed using LitServe with comprehensive benchmarking and optimization analysis.

## 📋 Overview

This project deploys a Cat-Dog classifier using:
- **Model**: ResNet18 (pretrained on ImageNet)
- **Framework**: LitServe + timm
- **GPU**: NVIDIA A10 (23GB VRAM)
- **Classification**: Aggregates ImageNet cat/dog class probabilities

## 📁 Files

| File | Description |
|------|-------------|
| `server.py` | LitServe API server (baseline, no batching) |
| `test_client.py` | Test client for verification |
| `benchmark.py` | Comprehensive benchmarking script |
| `test_cat.png` | Test cat image |
| `test_dog.png` | Test dog image |
| `benchmark_results.png` | Performance visualization |

## 🎯 Model Accuracy

The classifier correctly identifies cats and dogs with high confidence:

| Test | Prediction | Confidence | Top ImageNet Class |
|------|------------|------------|-------------------|
| 🐱 Cat Image | **CAT** | 99.95% | tabby_cat (40.62%) |
| 🐕 Dog Image | **DOG** | 100.00% | golden_retriever (99.58%) |

---

## 📊 Baseline Performance Results

### Theoretical Maximum (Direct GPU Inference)

| Batch Size | Throughput (reqs/sec) |
|------------|----------------------|
| 1          | 552.82               |
| 8          | 3,341.40             |
| 32         | 3,574.06             |
| 64         | **3,696.33** ⭐      |
| 128        | 3,653.47             |

**Key Finding**: GPU can process up to **3,696 requests/sec** with optimal batching (64 images).

### API Server Performance (Real-world, No Batching)

| Concurrency | Throughput | Efficiency | CPU Usage | GPU Usage | p95 Latency |
|-------------|------------|------------|-----------|-----------|-------------|
| 1           | 28.00 reqs/sec | 0.8% | 31.1% | 8.9% | 67.9ms |
| 8           | **41.71 reqs/sec** ⭐ | 1.1% | 33.0% | 3.3% | 222.4ms |
| 32          | 40.18 reqs/sec | 1.1% | 33.3% | 3.0% | 778.5ms |
| 64          | 38.63 reqs/sec | 1.0% | 40.7% | 3.4% | 1580.7ms |

---

## 🚨 Performance Analysis

### Key Findings

| Metric | Value |
|--------|-------|
| **Theoretical Maximum** | 3,696 reqs/sec |
| **Actual API Throughput** | 42 reqs/sec |
| **Efficiency** | **1.1%** (98.9% performance loss!) |
| **GPU Utilization** | ~3-9% (severely underutilized) |
| **CPU Utilization** | ~31-41% |

### 📈 Visualization

![Benchmark Results](benchmark_results.png)

The plots reveal:
1. **Top-Left**: Baseline model scales massively with batch size (553 → 3,696 reqs/sec)
2. **Top-Right**: API throughput is flat (no batching = no GPU benefit)
3. **Bottom-Left**: GPU usage is extremely low (~3-9%), indicating GPU is idle most of the time
4. **Bottom-Right**: Response time increases linearly with concurrency (requests queue up)

### 🔍 Root Cause Analysis

The **98.9% performance gap** exists because:

1. **No Batching**: Server processes ONE image at a time
   - GPU can handle 64 images in the same time as 1 image
   - Current: 42 reqs/sec → Potential: 3,696 reqs/sec

2. **CPU Bottleneck**: Image decoding and preprocessing is sequential
   - Base64 decode → requires CPU
   - Image transform → requires CPU
   - Only model inference uses GPU

3. **Single Worker**: Only one worker handles all requests
   - Requests queue up and wait
   - p95 latency increases dramatically (68ms → 1,581ms)

---

## 🚀 Optimization Roadmap

| Optimization | Expected Improvement | Status |
|--------------|---------------------|--------|
| **Baseline** | 42 reqs/sec (1.1%) | ✅ Complete |
| Enable Batching | ~200+ reqs/sec (~5%) | ⏳ Next |
| Add Workers (4) | ~400+ reqs/sec (~10%) | ⏳ Pending |
| Parallel Decoding | ~600+ reqs/sec (~16%) | ⏳ Pending |
| Half Precision (FP16) | ~1,000+ reqs/sec (~27%) | ⏳ Pending |
| **Fully Optimized** | ~2,000+ reqs/sec (54%+) | 🎯 Target |

---

## 🛠️ Usage

### Start Server
```bash
cd /home/ubuntu/EMLO4/s09/assignment/catdog
python server.py
```

### Test Classification
```bash
python test_client.py
```

### Run Benchmarks
```bash
python benchmark.py
```

---

## 📦 Requirements

```bash
pip install litserve timm torch pillow requests psutil gpustat matplotlib numpy
```

---

## 🎓 Assignment Checklist

- [x] Deploy Cat-Dog classifier with LitServe
- [x] Benchmark server performance
- [x] Compare with theoretical maximum throughput
- [x] Identify bottlenecks (GPU utilization only 3-9%!)
- [x] Document findings with plots
- [ ] Optimize incrementally (batching, workers, precision)
- [ ] Deploy LLM with LitServe
- [ ] Benchmark LLM tokens/sec
