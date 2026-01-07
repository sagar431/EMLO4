# 🦙 LLM Deployment with LitServe

Deployment of a Llama-based Large Language Model (LLM) using LitServe with OpenAI-compatible API.

## 📋 Overview

- **Model**: [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
  - Selected for stability and performance after dependency issues with GPT-OSS-20B.
  - Architecture: Llama-based
  - Precision: `bfloat16`
- **Framework**: LitServe + Transformers Pipeline
- **API Spec**: OpenAI-compatible (`/v1/chat/completions`)
- **Hardware**: NVIDIA A10 GPU

## 🚀 Performance Benchmarks

Benchmarks were run using varying prompt complexities to measure **Tokens Per Second (TPS)**.

| Metric | Result |
|--------|--------|
| **Average Throughput** | **56.37 tokens/sec** |
| **Peak Throughput** | **81.68 tokens/sec** |
| **Latency (avg)** | 2.80s |
| **Theoretical Max (Est)** | ~176 tokens/sec (Memory Bound) |
| **Utilization** | ~32-46% of theoretical max |

### 📊 Visualization
![Benchmark Results](benchmark_results.png)

### Detailed Breakdown

| Prompt Type | Avg TPS | Latency |
|-------------|---------|---------|
| Arithmetic (Short) | ~20 | 0.23s |
| Explanation (Medium) | ~81 | 2.06s |
| Coding (Medium) | ~48 | 2.34s |
| Comparison (Long) | ~67 | 4.71s |

*> Note: The variation in TPS is due to "Time to First Token" (TTFT) overhead dominating shorter generations. Longer generations amortize this overhead, reaching higher sustained TPS.*

---

## 🛠️ Usage

### 1. Requirements
This project uses a dedicated virtual environment to manage dependencies (torch, transformers, etc.).
```bash
# Activate the virtual environment
source .venv/bin/activate
```

### 2. Start the Server
```bash
python server.py
# Server runs on http://0.0.0.0:8001
```

### 3. Test the API (Client)
```bash
python test_client.py
```
This runs a conversation test using `openai` python client.

### 4. Run Benchmarks
```bash
python benchmark.py
```

### 5. Curl Example
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smollm2-1.7b",
    "messages": [
      { "role": "user", "content": "What is the capital of France?" }
    ]
  }'
```

---

## 📝 Implementation Details

- **`server.py`**: A LitServe `LitAPI` implementation using `transformers.pipeline`. 
  - Uses `OpenAISpec` for standardized API interface.
  - Implements streaming-capable generation logic.
- **`benchmark.py`**: Calculates TPS by measuring generation time relative to output length.

## 🔧 Troubleshooting

If you encounter `transformers` or `torch` errors, ensure you are in the correct virtual environment where compatible versions are installed.
```bash
source /home/ubuntu/EMLO4/s09/assignment/llm/.venv/bin/activate
```
