import socket
from contextlib import asynccontextmanager
from typing import List, Optional
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = None
tokenizer = None
device = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    pod: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, device_map="auto" if device.type == "cuda" else None)
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    print("Model loaded!")
    yield

app = FastAPI(title="TinyLlama Chat", lifespan=lifespan)
hostname = socket.gethostname()

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TinyLlama Chat</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            width: 100%;
            max-width: 800px;
            height: 90vh;
            display: flex;
            flex-direction: column;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        .header {
            padding: 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            text-align: center;
        }
        .header h1 {
            color: white;
            font-size: 1.8rem;
            font-weight: 700;
        }
        .header .pod-info {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.85rem;
            margin-top: 8px;
        }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            max-width: 80%;
            padding: 14px 18px;
            border-radius: 18px;
            font-size: 0.95rem;
            line-height: 1.5;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }
        .message.assistant {
            align-self: flex-start;
            background: rgba(255, 255, 255, 0.1);
            color: #e0e0e0;
            border-bottom-left-radius: 4px;
        }
        .input-container {
            padding: 20px 24px;
            background: rgba(0, 0, 0, 0.2);
            display: flex;
            gap: 12px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        #message-input {
            flex: 1;
            padding: 14px 20px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.05);
            color: white;
            font-size: 1rem;
            outline: none;
            transition: all 0.3s ease;
        }
        #message-input:focus {
            border-color: #667eea;
            background: rgba(255, 255, 255, 0.1);
        }
        #send-btn {
            padding: 14px 28px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        #send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        #send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .typing-indicator {
            display: flex;
            gap: 6px;
            padding: 14px 18px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 18px;
            width: fit-content;
        }
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #a78bfa;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }
        .typing-indicator span:nth-child(1) { animation-delay: 0s; }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1.2); opacity: 1; }
        }
        .welcome {
            text-align: center;
            color: rgba(255, 255, 255, 0.6);
            padding: 40px;
        }
        .welcome h2 {
            font-size: 1.5rem;
            margin-bottom: 10px;
            color: #a78bfa;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦙 TinyLlama Chat</h1>
            <div class="pod-info">Pod: HOSTNAME | Model: TinyLlama-1.1B-Chat</div>
        </div>
        <div class="chat-container" id="chat">
            <div class="welcome">
                <h2>👋 Hello!</h2>
                <p>I'm TinyLlama, a small but capable AI.<br>Ask me anything!</p>
            </div>
        </div>
        <div class="input-container">
            <input type="text" id="message-input" placeholder="Type your message...">
            <button id="send-btn" onclick="sendMessage()">Send 🚀</button>
        </div>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('message-input');
        const btn = document.getElementById('send-btn');
        let history = [];

        input.addEventListener('keypress', e => {
            if (e.key === 'Enter' && !btn.disabled) sendMessage();
        });

        function addMessage(role, content) {
            const welcome = document.querySelector('.welcome');
            if (welcome) welcome.remove();
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.textContent = content;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function addTyping() {
            const div = document.createElement('div');
            div.className = 'typing-indicator';
            div.id = 'typing';
            div.innerHTML = '<span></span><span></span><span></span>';
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function removeTyping() {
            const t = document.getElementById('typing');
            if (t) t.remove();
        }

        async function sendMessage() {
            const msg = input.value.trim();
            if (!msg) return;
            addMessage('user', msg);
            history.push({role: 'user', content: msg});
            input.value = '';
            btn.disabled = true;
            addTyping();
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({messages: history, max_new_tokens: 256, temperature: 0.7})
                });
                const data = await res.json();
                removeTyping();
                addMessage('assistant', data.response);
                history.push({role: 'assistant', content: data.response});
            } catch (e) {
                removeTyping();
                addMessage('assistant', 'Error occurred. Please try again.');
            }
            btn.disabled = false;
            input.focus();
        }
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML.replace("HOSTNAME", hostname)

@app.get("/health")
async def health():
    return {"status": "healthy", "pod": hostname}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    prompt = "<|system|>\nYou are a helpful assistant.</s>\n"
    for m in request.messages:
        if m.role == "user":
            prompt += f"<|user|>\n{m.content}</s>\n"
        else:
            prompt += f"<|assistant|>\n{m.content}</s>\n"
    prompt += "<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=request.max_new_tokens, temperature=request.temperature, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = generated.split("<|assistant|>\n")[-1].replace("</s>", "").strip()
    return ChatResponse(response=response, pod=hostname)
