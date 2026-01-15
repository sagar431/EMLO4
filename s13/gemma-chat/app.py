import socket
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global model and tokenizer
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    
    if device.type == "cpu":
        model = model.to(device)
    
    model.eval()
    print("Model loaded successfully!")
    
    yield
    print("Shutting down...")


app = FastAPI(
    title="TinyLlama Chat API",
    description="Chat with TinyLlama-1.1B-Chat model",
    version="1.0.0",
    lifespan=lifespan,
)

hostname = socket.gethostname()

HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TinyLlama Chat</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: system-ui, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
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
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
        }
        .header {
            padding: 24px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            text-align: center;
            color: white;
        }
        .header h1 { font-size: 1.8rem; }
        .header .pod-info { opacity: 0.7; font-size: 0.85rem; margin-top: 8px; }
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
        }
        .message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .message.assistant {
            align-self: flex-start;
            background: rgba(255,255,255,0.1);
            color: #e0e0e0;
        }
        .input-container {
            padding: 20px;
            background: rgba(0,0,0,0.2);
            display: flex;
            gap: 12px;
        }
        #message-input {
            flex: 1;
            padding: 14px;
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            background: rgba(255,255,255,0.05);
            color: white;
            font-size: 1rem;
            outline: none;
        }
        #message-input:focus { border-color: #667eea; }
        #send-btn {
            padding: 14px 28px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: bold;
            cursor: pointer;
        }
        #send-btn:disabled { opacity: 0.5; }
        .welcome { text-align: center; color: rgba(255,255,255,0.6); padding: 40px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>TinyLlama Chat</h1>
            <div class="pod-info">Pod: HOSTNAME | Model: TinyLlama-1.1B-Chat</div>
        </div>
        <div class="chat-container" id="chat">
            <div class="welcome"><h2>Hello!</h2><p>I'm TinyLlama. Ask me anything!</p></div>
        </div>
        <div class="input-container">
            <input type="text" id="message-input" placeholder="Type your message...">
            <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('message-input');
        const btn = document.getElementById('send-btn');
        let history = [];
        
        input.addEventListener('keypress', e => { if (e.key === 'Enter') sendMessage(); });
        
        function addMsg(role, content) {
            const welcome = document.querySelector('.welcome');
            if (welcome) welcome.remove();
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.textContent = content;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        async function sendMessage() {
            const msg = input.value.trim();
            if (!msg) return;
            addMsg('user', msg);
            history.push({role: 'user', content: msg});
            input.value = '';
            btn.disabled = true;
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({messages: history, max_new_tokens: 256, temperature: 0.7})
                });
                const data = await res.json();
                addMsg('assistant', data.response);
                history.push({role: 'assistant', content: data.response});
            } catch (e) {
                addMsg('assistant', 'Error occurred. Please try again.');
            }
            btn.disabled = false;
            input.focus();
        }
    </script>
</body>
</html>'''


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE.replace("HOSTNAME", hostname)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    prompt = "<|system|>\nYou are a helpful AI assistant.</s>\n"
    for msg in request.messages:
        if msg.role == "user":
            prompt += "
