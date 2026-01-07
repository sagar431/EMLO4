"""
Test client for GPT-OSS-20B LLM Server
Uses OpenAI-compatible API
"""

from openai import OpenAI


def test_chat():
    """Test the LLM with a simple chat"""
    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="dummy-key"  # Not needed for local server
    )
    
    print("=" * 60)
    print("GPT-OSS-20B Chat Test")
    print("=" * 60)
    
    # Test 1: Simple question
    print("\n📝 Test 1: Simple Question")
    print("-" * 40)
    
    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[
            {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
        ],
        stream=False
    )
    
    print(f"Response: {response.choices[0].message.content}")
    
    # Test 2: Coding question
    print("\n📝 Test 2: Coding Question")
    print("-" * 40)
    
    response = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[
            {"role": "user", "content": "Write a Python function to calculate fibonacci numbers. Keep it short."}
        ],
        stream=False
    )
    
    print(f"Response: {response.choices[0].message.content}")
    
    print("\n" + "=" * 60)
    print("✅ Tests completed!")
    print("=" * 60)


def test_streaming():
    """Test streaming response"""
    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="dummy-key"
    )
    
    print("\n📝 Streaming Test")
    print("-" * 40)
    
    stream = client.chat.completions.create(
        model="gpt-oss-20b",
        messages=[
            {"role": "user", "content": "Explain quantum computing in 3 sentences."}
        ],
        stream=True
    )
    
    print("Response: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    test_chat()
