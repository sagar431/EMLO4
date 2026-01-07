"""
SmolLM2-1.7B LLM Server using LitServe

Deploys a Llama-based model using pipeline API for better compatibility.
No batching as per assignment requirements.
"""

import torch
from transformers import pipeline
import litserve as ls
import time


class SmolLMAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the model using pipeline"""
        self.device = device
        model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        
        print(f"Loading {model_id}...")
        print(f"Device: {device}")
        
        # Use pipeline for simpler loading
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device=device if device != "cuda:0" else 0,
        )
        
        print(f"Model loaded successfully!")
        
    def decode_request(self, request):
        """Process incoming chat request"""
        if not request.messages:
            raise ValueError("No messages provided")
        
        # Convert to list of dicts
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        return messages
    
    def predict(self, messages, context):
        """Generate response and measure tokens/sec"""
        start_time = time.perf_counter()
        
        outputs = self.pipe(
            messages,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        end_time = time.perf_counter()
        
        # Extract generated text
        generated = outputs[0]["generated_text"]
        # Get just the assistant response (last message)
        if isinstance(generated, list):
            response_text = generated[-1]["content"] if generated else ""
        else:
            response_text = str(generated)
        
        # Estimate tokens (rough approximation: 4 chars per token)
        estimated_tokens = len(response_text) // 4
        generation_time = end_time - start_time
        tokens_per_sec = estimated_tokens / generation_time if generation_time > 0 else 0
        
        yield {
            "response": response_text,
            "tokens_generated": estimated_tokens,
            "generation_time": generation_time,
            "tokens_per_sec": tokens_per_sec
        }
    
    def encode_response(self, outputs):
        """Format the response"""
        for output in outputs:
            yield {
                "role": "assistant",
                "content": output["response"],
                "usage": {
                    "tokens_generated": output["tokens_generated"],
                    "generation_time_sec": round(output["generation_time"], 3),
                    "tokens_per_sec": round(output["tokens_per_sec"], 2)
                }
            }


if __name__ == "__main__":
    api = SmolLMAPI()
    
    # Server with OpenAI-compatible API spec
    # No batching as per assignment requirements
    server = ls.LitServer(
        api,
        spec=ls.OpenAISpec(),
        accelerator="gpu",
        workers_per_device=1,  # Single worker, no batching
    )
    server.run(port=8001)
