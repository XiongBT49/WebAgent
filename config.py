"""Configuration file for the VLLM Web Agent"""

import os
from dotenv import load_dotenv

# Load environment variables from a .env file (for API keys)
# Create a file named .env in this directory and add:
# DEEPSEEK_API_KEY="your_actual_api_key"
load_dotenv()

# ------------------------------------------------------------------------------
# LLM Provider Configuration
# ------------------------------------------------------------------------------
# Choose your LLM provider: 'ollama' or 'deepseek'
LLM_PROVIDER = "ollama"  # 使用本地Ollama的Qwen模型
# LLM_PROVIDER = "deepseek"  # 备选：云端DeepSeek

# --- Ollama Configuration (Local) ---
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:7b"  # 文本推理模型

# --- DeepSeek API Configuration (Cloud) ---
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
# DEEPSEEK_MODEL = "deepseek-reasoner"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# --- Vision Model Configuration (for screenshot analysis) ---
# Enable vision model to analyze screenshots alongside DOM
USE_VISION_MODEL = True

# Use Ollama with Qwen2.5-VL-32B (Recommended - powerful local vision model)
VISION_MODEL_PROVIDER = "ollama"
OLLAMA_VISION_MODEL = "qwen2.5vl:32b"  # Qwen 32B vision model via Ollama

# Alternative: Use vLLM for Qwen2-VL (requires separate server)
# VISION_MODEL_PROVIDER = "qwen-vl-local"
# QWEN_VL_BASE_URL = "http://localhost:8000/v1"
# QWEN_VL_MODEL = "Qwen2-VL-32B-Instruct"
# 启动命令: bash start_qwen_vl.sh

# --- GPT-4 Vision (Cloud, commented out) ---
# VISION_MODEL_PROVIDER = "gpt-4-vision"
# VISION_MODEL_NAME = "gpt-4-vision-preview"
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------------------------------------
# Agent & Browser Configuration
# ------------------------------------------------------------------------------
# Set to True to run the browser in headless mode (no GUI)
BROWSER_HEADLESS = True

# Default timeout for browser operations (in milliseconds)
BROWSER_TIMEOUT = 30000

# The maximum number of iterations the agent can perform for a task
MAX_ITERATIONS = 20

# Model temperature (0.0 for deterministic results, 0.7 for more creative)
TEMPERATURE = 0.0

# ------------------------------------------------------------------------------
# Output Configuration
# ------------------------------------------------------------------------------
# Directory to save logs, screenshots, and other outputs
OUTPUT_DIR = "output"
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
SCREENSHOTS_DIR = os.path.join(OUTPUT_DIR, "screenshots")
PDFS_DIR = os.path.join(OUTPUT_DIR, "pdfs")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
