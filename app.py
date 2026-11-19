"""Mochi-103M Web Studio
A tactile, elegant web interface for Mochi-103M.
Features Key-Value (KV) Caching, Min-P sampling, Chain-of-Thought collapsible reasoning,
and real-time local GPU inference with zero external pip dependencies.

Usage:
    python app.py [--port 8000] [--device auto]
"""
import argparse
import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.parse

import torch
import torch.nn.functional as F

from config import ModelConfig
from model import GPT
from tokenizer import EOT_TOKEN, load_tokenizer
from utils import load_checkpoint, strip_state_dict_prefixes

# Global state
LOADED_MODELS = {}
GLOBAL_TOKENIZER = None
DEVICE = None

def get_device(pref="auto"):
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)

def load_mochi_model(model_key="reasoning", force_reload=False):
    global LOADED_MODELS, GLOBAL_TOKENIZER, DEVICE
    if not force_reload and model_key in LOADED_MODELS:
        return LOADED_MODELS[model_key]
    
    ckpt_map = {
        "reasoning": "out/reasoning/best.pt",
        "chat": "out/chat/best.pt",
        "base": "out/base/best.pt"
    }
    ckpt_path = ckpt_map.get(model_key, "out/reasoning/best.pt")
    if not os.path.exists(ckpt_path):
        if os.path.exists("out/reasoning/best.pt"):
            ckpt_path = "out/reasoning/best.pt"
        elif os.path.exists("out/chat/best.pt"):
            ckpt_path = "out/chat/best.pt"
        elif os.path.exists("out/base/best.pt"):
            ckpt_path = "out/base/best.pt"

    print(f"Loading {model_key} from {ckpt_path} on {DEVICE}...")
    checkpoint = load_checkpoint(ckpt_path, device=str(DEVICE))
    config = ModelConfig(**checkpoint["model_config"])
    model = GPT(config).to(DEVICE)
    model.load_state_dict(strip_state_dict_prefixes(checkpoint["model"]))
    model.eval()
    
    LOADED_MODELS[model_key] = (model, checkpoint.get("reasoning", False), checkpoint.get("sft", False))
    return LOADED_MODELS[model_key]

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mochi-103M Studio</title>
<style>
:root {
  --bg-page: #f8fafc;
  --bg-sidebar: #ffffff;
  --border-light: #e2e8f0;
  --border-dark: #cbd5e1;
  --text-main: #0f172a;
  --text-muted: #64748b;
  --primary: #0284c7;
  --primary-gradient: linear-gradient(180deg, #0284c7 0%, #0369a1 100%);
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
  --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
}

* { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
body { background: var(--bg-page); color: var(--text-main); height: 100vh; display: flex; overflow: hidden; }

/* Minimalist Sidebar */
.sidebar {
  width: 320px;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border-light);
  display: flex;
  flex-direction: column;
  padding: 1.25rem;
  gap: 1.1rem;
  overflow-y: auto;
  flex-shrink: 0;
}

.logo-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--border-light);
}
.logo-title {
  font-size: 1.15rem;
  font-weight: 800;
  color: var(--text-main);
  letter-spacing: -0.02em;
}
.logo-badge {
  font-size: 0.68rem;
  background: #f0f9ff;
  color: var(--primary);
  border: 1px solid #bae6fd;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 5px;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}
.control-label {
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.control-value {
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--primary);
  background: #f1f5f9;
  padding: 1px 6px;
  border-radius: 4px;
}

select {
  width: 100%;
  background: #ffffff;
  color: var(--text-main);
  border: 1px solid var(--border-light);
  padding: 0.6rem 0.75rem;
  border-radius: 8px;
  font-size: 0.88rem;
  font-weight: 600;
  outline: none;
  cursor: pointer;
  transition: all 0.15s ease;
  box-shadow: var(--shadow-sm);
}
select:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.12);
}

/* Collapsible Settings Drawer */
.settings-drawer {
  border: 1px solid var(--border-light);
  border-radius: 8px;
  background: #ffffff;
  overflow: hidden;
  transition: all 0.2s ease;
}
.settings-summary {
  padding: 0.65rem 0.85rem;
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--text-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  user-select: none;
  background: #ffffff;
}
.settings-summary:hover {
  background: #f8fafc;
  color: var(--text-main);
}
.settings-drawer[open] .settings-summary {
  border-bottom: 1px solid var(--border-light);
  background: #f8fafc;
}
.settings-content {
  padding: 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  background: #ffffff;
}

.mode-row {
  display: flex;
  gap: 0.35rem;
  background: #f1f5f9;
  padding: 3px;
  border-radius: 6px;
}
.mode-btn {
  flex: 1;
  background: transparent;
  border: none;
  font-size: 0.74rem;
  font-weight: 700;
  color: var(--text-muted);
  padding: 5px 0;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s ease;
}
.mode-btn.active {
  background: #ffffff;
  color: var(--primary);
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

input[type=range] {
  -webkit-appearance: none;
  width: 100%;
  background: transparent;
}
input[type=range]:focus { outline: none; }
input[type=range]::-webkit-slider-runnable-track {
  width: 100%;
  height: 5px;
  cursor: pointer;
  background: #e2e8f0;
  border-radius: 3px;
}
input[type=range]::-webkit-slider-thumb {
  height: 15px;
  width: 15px;
  border-radius: 50%;
  background: #ffffff;
  cursor: pointer;
  -webkit-appearance: none;
  margin-top: -5px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
  border: 2px solid var(--primary);
}

/* Minimal Hardware Card */
.hardware-card {
  margin-top: auto;
  background: #f8fafc;
  border: 1px solid var(--border-light);
  border-radius: 8px;
  padding: 0.85rem;
  font-size: 0.76rem;
  color: var(--text-muted);
  line-height: 1.5;
}
.hardware-header {
  font-size: 0.78rem;
  font-weight: 800;
  color: var(--text-main);
  margin-bottom: 4px;
}
.hw-table {
  width: 100%;
  border-collapse: collapse;
}
.hw-table td {
  padding: 2px 0;
}
.hw-label {
  color: #64748b;
  width: 38%;
  font-weight: 600;
}
.hw-val {
  color: #0f172a;
  font-weight: 700;
}

/* Main Chat Section */
.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg-page);
  position: relative;
}

.top-bar {
  height: 56px;
  background: #ffffff;
  border-bottom: 1px solid var(--border-light);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 1.5rem;
}
.bar-info {
  display: flex;
  gap: 0.6rem;
  align-items: center;
  font-size: 0.84rem;
  font-weight: 600;
  color: var(--text-muted);
}
.status-pill {
  font-size: 0.68rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 9999px;
  background: #ecfdf5;
  color: #059669;
  border: 1px solid #a7f3d0;
}

.clear-btn {
  background: #ffffff;
  border: 1px solid var(--border-light);
  color: var(--text-muted);
  font-size: 0.78rem;
  font-weight: 600;
  padding: 5px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s ease;
}
.clear-btn:hover { background: #f8fafc; color: var(--text-main); border-color: var(--border-dark); }

/* Messages Area */
.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 2rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
  max-width: 860px;
  width: 100%;
  margin: 0 auto;
}

.msg {
  display: flex;
  flex-direction: column;
  max-width: 84%;
  animation: popIn 0.18s cubic-bezier(0.16, 1, 0.3, 1);
}
.msg.user { align-self: flex-end; }
.msg.assistant { align-self: flex-start; }

.msg-card {
  padding: 0.95rem 1.2rem;
  border-radius: 14px;
  font-size: 0.94rem;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
}
.msg.user .msg-card {
  background: #0f172a;
  color: #ffffff;
  border-bottom-right-radius: 4px;
  box-shadow: var(--shadow-sm);
}
.msg.assistant .msg-card {
  background: #ffffff;
  color: #1e293b;
  border: 1px solid var(--border-light);
  border-bottom-left-radius: 4px;
  box-shadow: var(--shadow-md);
}
.msg-meta {
  font-size: 0.72rem;
  font-weight: 600;
  color: #94a3b8;
  margin-top: 4px;
  padding: 0 4px;
}

/* Prompt Starter Pills */
.quick-prompts {
  margin-top: 0.9rem;
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}
.quick-title {
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.pill-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}
.prompt-pill {
  background: #ffffff;
  border: 1px solid var(--border-light);
  color: var(--primary);
  font-size: 0.8rem;
  font-weight: 600;
  padding: 6px 12px;
  border-radius: 9999px;
  cursor: pointer;
  transition: all 0.15s ease;
  box-shadow: var(--shadow-sm);
}
.prompt-pill:hover {
  background: #f0f9ff;
  border-color: #bae6fd;
  color: #0369a1;
  transform: translateY(-1px);
}

/* Code Styling */
pre.code-block {
  background: #0f172a;
  color: #f8fafc;
  border-radius: 8px;
  padding: 0.85rem 1.1rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.86rem;
  overflow-x: auto;
  margin: 0.6rem 0;
  line-height: 1.5;
  border: 1px solid #1e293b;
}
code.inline-code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.88em;
  background: #f1f5f9;
  color: #0284c7;
  padding: 2px 6px;
  border-radius: 4px;
  border: 1px solid #e2e8f0;
}

/* Collapsible Chain-of-Thought Box */
.think-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 0.6rem 0.8rem;
  margin-bottom: 0.65rem;
  font-size: 0.84rem;
  color: #475569;
}
.think-header {
  font-weight: 700;
  color: var(--primary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  user-select: none;
}
.think-content {
  margin-top: 0.5rem;
  border-top: 1px dashed #cbd5e1;
  padding-top: 0.5rem;
  font-style: normal;
  display: none;
}
.think-content.open { display: block; }

/* Input Dock */
.input-area {
  padding: 1rem 1.5rem 1.5rem;
  background: linear-gradient(180deg, rgba(248,250,252,0) 0%, rgba(248,250,252,0.95) 40%, #f8fafc 100%);
  width: 100%;
}
.input-dock {
  max-width: 860px;
  margin: 0 auto;
  background: #ffffff;
  border: 1px solid var(--border-light);
  border-radius: 12px;
  padding: 0.5rem 0.6rem 0.5rem 0.9rem;
  display: flex;
  gap: 0.6rem;
  align-items: flex-end;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.input-dock:focus-within {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.12), 0 4px 16px rgba(0, 0, 0, 0.05);
}
textarea {
  flex: 1;
  background: transparent;
  border: none;
  color: var(--text-main);
  font-size: 0.94rem;
  padding: 0.4rem 0;
  resize: none;
  height: 42px;
  max-height: 150px;
  outline: none;
  line-height: 1.45;
}
.send-btn {
  background: var(--primary-gradient);
  color: #ffffff;
  border: none;
  border-radius: 8px;
  font-size: 0.88rem;
  font-weight: 700;
  padding: 0 1.25rem;
  height: 40px;
  cursor: pointer;
  transition: all 0.15s ease;
}
.send-btn:hover {
  filter: brightness(1.06);
}
.send-btn:disabled {
  background: #e2e8f0;
  color: #94a3b8;
  cursor: not-allowed;
}

@keyframes popIn {
  from { opacity: 0; transform: scale(0.98) translateY(4px); }
  to { opacity: 1; transform: scale(1) translateY(0); }
}
</style>
</head>
<body>

<!-- Minimalist Sidebar -->
<div class="sidebar">
  <div class="logo-header">
    <div class="logo-title">Mochi-103M</div>
    <span class="logo-badge">103M CAUSAL LM</span>
  </div>

  <div class="control-group">
    <div class="control-label"><span>Active Model</span></div>
    <select id="model-select">
      <option value="reasoning">Mochi-Reason (Chain-of-Thought)</option>
      <option value="chat">Mochi-Chat (Instruct SFT)</option>
      <option value="base">Mochi-Base (Pretrained)</option>
    </select>
  </div>

  <!-- Collapsible Settings Drawer -->
  <details class="settings-drawer">
    <summary class="settings-summary">
      <span style="display:flex; align-items:center; gap:6px;">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        Inference Settings
      </span>
      <span style="font-size:0.75rem;">&#9662;</span>
    </summary>
    <div class="settings-content">
      <div class="control-group">
        <div class="control-label"><span>Decoding Profile</span></div>
        <div class="mode-row">
          <button class="mode-btn active" onclick="setMode('accurate', event)">Precise</button>
          <button class="mode-btn" onclick="setMode('balanced', event)">Balanced</button>
          <button class="mode-btn" onclick="setMode('creative', event)">Creative</button>
        </div>
      </div>
      <div class="control-group">
        <div class="control-label"><span>Temperature</span><span class="control-value" id="temp-val">0.00</span></div>
        <input type="range" id="temp-slider" min="0.00" max="1.20" step="0.05" value="0.00">
      </div>
      <div class="control-group">
        <div class="control-label"><span>Min-P Threshold</span><span class="control-value" id="minp-val">0.08</span></div>
        <input type="range" id="minp-slider" min="0.00" max="0.30" step="0.01" value="0.08">
      </div>
      <div class="control-group">
        <div class="control-label"><span>Top-P Nucleus</span><span class="control-value" id="topp-val">0.85</span></div>
        <input type="range" id="topp-slider" min="0.10" max="1.00" step="0.05" value="0.85">
      </div>
      <div class="control-group">
        <div class="control-label"><span>Repetition Penalty</span><span class="control-value" id="rep-val">1.15</span></div>
        <input type="range" id="rep-slider" min="1.00" max="2.00" step="0.05" value="1.15">
      </div>
      <div class="control-group">
        <div class="control-label"><span>Max Response Tokens</span><span class="control-value" id="max-val">256</span></div>
        <input type="range" id="max-slider" min="32" max="512" step="32" value="256">
      </div>
    </div>
  </details>

  <!-- Hardware Platform Card -->
  <div class="hardware-card">
    <div class="hardware-header">Hardware Platform</div>
    <table class="hw-table">
      <tr><td class="hw-label">GPU</td><td class="hw-val">NVIDIA RTX 4050 Laptop (6GB)</td></tr>
      <tr><td class="hw-label">CPU</td><td class="hw-val">Intel Core i7-13620H</td></tr>
      <tr><td class="hw-label">Params</td><td class="hw-val">102,937,344 (103M)</td></tr>
      <tr><td class="hw-label">Layers</td><td class="hw-val">11 Layers &bull; 12 Heads &bull; 768 Dim</td></tr>
      <tr><td class="hw-label">Inference</td><td class="hw-val">Autoregressive KV-Cache O(T)</td></tr>
    </table>
  </div>
</div>

<!-- Main Chat View -->
<div class="main">
  <div class="top-bar">
    <div class="bar-info">
      <span>Hardware: <strong>NVIDIA GeForce RTX 4050 6GB Laptop GPU</strong></span>
      <span class="status-pill">AIR-GAPPED LOCAL INFERENCE</span>
    </div>
    <div>
      <button class="clear-btn" onclick="clearChat()">Clear Chat</button>
    </div>
  </div>

  <!-- Messages List -->
  <div class="messages-container" id="messages-container">
    <div class="msg assistant">
      <div class="msg-card">
        Hello! I am Mochi-103M, an autoregressive 103-million parameter causal language model trained from scratch on consumer hardware.
        <div class="quick-prompts">
          <div class="quick-title">Explore Domain Knowledge:</div>
          <div class="pill-grid">
            <button class="prompt-pill" onclick="sendQuickPrompt('What is a Transformer?')">What is a Transformer?</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('How does a Transformer work?')">How does a Transformer work?</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('What is self-attention?')">Self-Attention</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('What is RoPE?')">RoPE Mathematics</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('What is the difference between RMSNorm and LayerNorm?')">RMSNorm vs LayerNorm</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('What is SwiGLU and why is it used?')">SwiGLU Gating</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('How does Key-Value (KV) caching work?')">KV-Cache Complexity</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('What is FlashAttention?')">FlashAttention Tiling</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('Explain the Backpropagation algorithm.')">Backpropagation</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('Who is the President of India?')">President of India</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('Write Python code for binary search.')">Binary Search</button>
          </div>
        </div>
      </div>
      <div class="msg-meta">System Ready | KV-Cache Active | RTX 4050 CUDA</div>
    </div>
  </div>

  <!-- Input Dock -->
  <div class="input-area">
    <div class="input-dock">
      <textarea id="prompt-input" placeholder="Ask Mochi-103M about Deep Learning, Transformers, algorithms... (Enter to send)"></textarea>
      <button class="send-btn" id="send-btn" onclick="sendMessage()">Send</button>
    </div>
  </div>
</div>

<script>
const tempSlider = document.getElementById('temp-slider');
const tempVal = document.getElementById('temp-val');
tempSlider.oninput = () => tempVal.innerText = Number(tempSlider.value).toFixed(2);

const minpSlider = document.getElementById('minp-slider');
const minpVal = document.getElementById('minp-val');
minpSlider.oninput = () => minpVal.innerText = Number(minpSlider.value).toFixed(2);

const toppSlider = document.getElementById('topp-slider');
const toppVal = document.getElementById('topp-val');
toppSlider.oninput = () => toppVal.innerText = Number(toppSlider.value).toFixed(2);

const repSlider = document.getElementById('rep-slider');
const repVal = document.getElementById('rep-val');
repSlider.oninput = () => repVal.innerText = Number(repSlider.value).toFixed(2);

const maxSlider = document.getElementById('max-slider');
const maxVal = document.getElementById('max-val');
maxSlider.oninput = () => maxVal.innerText = maxSlider.value;

function setMode(mode, e) {
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  e.target.classList.add('active');
  if (mode === 'accurate') {
    tempSlider.value = 0.00;
    tempVal.innerText = "0.00";
    repSlider.value = 1.10;
    repVal.innerText = "1.10";
  } else if (mode === 'balanced') {
    tempSlider.value = 0.30;
    tempVal.innerText = "0.30";
    repSlider.value = 1.15;
    repVal.innerText = "1.15";
  } else if (mode === 'creative') {
    tempSlider.value = 0.70;
    tempVal.innerText = "0.70";
    repSlider.value = 1.20;
    repVal.innerText = "1.20";
  }
}

const inputField = document.getElementById('prompt-input');
inputField.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
inputField.addEventListener('input', () => {
  inputField.style.height = 'auto';
  inputField.style.height = Math.min(inputField.scrollHeight, 150) + 'px';
});

function sendQuickPrompt(promptText) {
  inputField.value = promptText;
  sendMessage();
}

function clearChat() {
  const container = document.getElementById('messages-container');
  container.innerHTML = `
    <div class="msg assistant">
      <div class="msg-card">
        Conversation cleared. What would you like to explore next?
        <div class="quick-prompts">
          <div class="quick-title">Explore Domain Knowledge:</div>
          <div class="pill-grid">
            <button class="prompt-pill" onclick="sendQuickPrompt('What is a Transformer?')">What is a Transformer?</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('How does a Transformer work?')">How does a Transformer work?</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('What is self-attention?')">Self-Attention</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('What is RoPE?')">RoPE Mathematics</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('What is the difference between RMSNorm and LayerNorm?')">RMSNorm vs LayerNorm</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('What is SwiGLU and why is it used?')">SwiGLU Gating</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('How does Key-Value (KV) caching work?')">KV-Cache Complexity</button>
            <button class="prompt-pill" onclick="sendQuickPrompt('Who is the President of India?')">President of India</button>
          </div>
        </div>
      </div>
      <div class="msg-meta">System Ready | KV-Cache Active | RTX 4050 CUDA</div>
    </div>
  `;
}

async function sendMessage() {
  const text = inputField.value.trim();
  if (!text) return;

  const container = document.getElementById('messages-container');
  const sendBtn = document.getElementById('send-btn');
  
  // 1. Append User Message
  const userDiv = document.createElement('div');
  userDiv.className = 'msg user';
  userDiv.innerHTML = `<div class="msg-card">${escapeHtml(text)}</div>`;
  container.appendChild(userDiv);

  inputField.value = '';
  inputField.style.height = '42px';
  sendBtn.disabled = true;
  container.scrollTop = container.scrollHeight;

  // 2. Append Assistant Loading Bubble
  const asstDiv = document.createElement('div');
  asstDiv.className = 'msg assistant';
  const bubble = document.createElement('div');
  bubble.className = 'msg-card';
  bubble.innerText = 'Computing forward pass on RTX 4050 GPU...';
  asstDiv.appendChild(bubble);
  container.appendChild(asstDiv);
  container.scrollTop = container.scrollHeight;

  const payload = {
    model: document.getElementById('model-select').value,
    prompt: text,
    temperature: parseFloat(tempSlider.value),
    min_p: parseFloat(minpSlider.value),
    top_p: parseFloat(toppSlider.value),
    repetition_penalty: parseFloat(repSlider.value),
    max_new_tokens: parseInt(maxSlider.value)
  };

  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await response.json();

    let fullText = data.text || '';
    let thinkContent = '';
    
    // Check if response contains <think> tag
    if (fullText.includes('<think>') && fullText.includes('</think>')) {
      const parts = fullText.split('</think>');
      thinkContent = parts[0].replace('<think>', '').trim();
      fullText = parts[1].trim();
    } else if (fullText.startsWith("Let's think step by step:")) {
      const lines = fullText.split("\\nTherefore");
      if (lines.length > 1) {
        thinkContent = lines[0].replace("Let's think step by step:", '').trim();
        fullText = "Therefore" + lines[1];
      }
    }

    bubble.innerHTML = '';

    if (thinkContent) {
      const thinkBox = document.createElement('div');
      thinkBox.className = 'think-card';
      thinkBox.innerHTML = `
        <div class="think-header" onclick="this.nextElementSibling.classList.toggle('open')">
          <span>Reasoning Trace</span>
          <span style="font-size:0.75rem;">Toggle</span>
        </div>
        <div class="think-content open">${escapeHtml(thinkContent)}</div>
      `;
      bubble.appendChild(thinkBox);
    }

    const textDiv = document.createElement('div');
    textDiv.innerHTML = formatMarkdown(fullText);
    bubble.appendChild(textDiv);

    const metaDiv = document.createElement('div');
    metaDiv.className = 'msg-meta';
    metaDiv.innerHTML = `Generated ${data.tokens_generated} tokens in ${data.latency_sec}s (${data.tokens_per_sec} tok/s) | KV-Cache O(T) | RTX 4050 CUDA`;
    asstDiv.appendChild(metaDiv);

  } catch (err) {
    bubble.innerText = 'Error: ' + err.message;
  } finally {
    sendBtn.disabled = false;
    inputField.focus();
    container.scrollTop = container.scrollHeight;
  }
}

function formatMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/```(?:[a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g, (match, code) => {
    return `<pre class="code-block"><code>${code.trim()}</code></pre>`;
  });
  html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
  return html;
}

function escapeHtml(text) {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}
</script>
</body>
</html>
"""

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class StudioHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            req = json.loads(body.decode("utf-8"))

            model_key = req.get("model", "reasoning")
            prompt = req.get("prompt", "")
            temperature = float(req.get("temperature", 0.0))
            min_p = float(req.get("min_p", 0.08))
            top_p = float(req.get("top_p", 0.85))
            rep_pen = float(req.get("repetition_penalty", 1.15))
            max_new = int(req.get("max_new_tokens", 256))

            model, is_reasoning, is_sft = load_mochi_model(model_key, force_reload=False)
            is_instruct = is_reasoning or is_sft

            clean_prompt = prompt.strip()
            if is_instruct:
                formatted_prompt = f"### Instruction:\n{clean_prompt}\n\n### Response:\n"
            else:
                formatted_prompt = f"Question: {clean_prompt}\nAnswer:"

            eot_id = GLOBAL_TOKENIZER.token_to_id(EOT_TOKEN)
            input_ids = GLOBAL_TOKENIZER.encode(formatted_prompt).ids
            idx = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
            block_size = model.config.block_size

            generated_ids = []
            generated_text = ""
            past_kv = None
            t0 = time.perf_counter()

            with torch.no_grad():
                for step in range(max_new):
                    if step == 0:
                        logits, past_kv = model(idx[:, -block_size:], use_cache=True)
                    else:
                        logits, past_kv = model(next_token, past_kv=past_kv, use_cache=True)
                    
                    logits = logits[:, -1, :]

                    # 3-gram repetition blocking (prevents infinite repetitive loops)
                    if len(generated_ids) >= 3:
                        recent = tuple(generated_ids[-2:])
                        for i in range(len(generated_ids) - 2):
                            if (generated_ids[i], generated_ids[i+1]) == recent:
                                banned = generated_ids[i+2]
                                logits[0, banned] = float("-inf")

                    # Smart repetition penalty: only penalize tokens repeating >= 2 times
                    if rep_pen != 1.0 and generated_ids:
                        counts = {}
                        for tid in generated_ids:
                            counts[tid] = counts.get(tid, 0) + 1
                        for tid, c in counts.items():
                            if c >= 2:
                                pen = rep_pen ** min(c - 1, 3)
                                if logits[0, tid] > 0:
                                    logits[0, tid] /= pen
                                else:
                                    logits[0, tid] *= pen

                    if temperature == 0:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    else:
                        logits = logits / temperature
                        kth_val = torch.topk(logits, min(40, logits.size(-1)))[0][:, -1, None]
                        logits = logits.masked_fill(logits < kth_val, float("-inf"))
                        if top_p < 1.0:
                            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            remove = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
                            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                            logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                        if min_p > 0.0:
                            probs_all = F.softmax(logits, dim=-1)
                            p_max = torch.max(probs_all, dim=-1, keepdim=True)[0]
                            logits = logits.masked_fill(probs_all < (p_max * min_p), float("-inf"))
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                    tid = next_token.item()
                    if tid == eot_id:
                        break

                    t_str = GLOBAL_TOKENIZER.decode([tid])
                    if ("###" in t_str or "\n\nUser:" in t_str) and len(generated_text) > 10:
                        break

                    generated_ids.append(tid)
                    generated_text += t_str

            elapsed = time.perf_counter() - t0
            tok_count = len(generated_ids)
            tok_speed = round(tok_count / elapsed, 1) if elapsed > 0 else 0

            resp_payload = {
                "text": generated_text.strip(),
                "tokens_generated": tok_count,
                "latency_sec": round(elapsed, 2),
                "tokens_per_sec": tok_speed
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default server logs for quiet output

def main():
    global GLOBAL_TOKENIZER, DEVICE
    parser = argparse.ArgumentParser(description="Mochi-103M Web Studio")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--device", default="auto", help="Compute device (auto/cuda/cpu)")
    args = parser.parse_args()

    DEVICE = get_device(args.device)
    print(f"Initializing Mochi-103M Studio on {DEVICE}...")

    tok_path = "out/reasoning/tokenizer.json"
    if not os.path.exists(tok_path):
        tok_path = "data/tokenizer.json"
    GLOBAL_TOKENIZER = load_tokenizer(tok_path)
    print("Tokenizer initialized.")

    # Pre-warm default model
    load_mochi_model("reasoning")

    server_address = ("127.0.0.1", args.port)
    httpd = ThreadedHTTPServer(server_address, StudioHandler)
    print(f"\nMochi-103M Web Studio live at: http://localhost:{args.port}")
    print("Press Ctrl+C to terminate.")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    main()
