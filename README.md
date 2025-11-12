# WebAgent - Vision-Language Web Agent

一个基于视觉语言模型的智能网页自动化代理,能够理解图片内容并自主完成复杂的网页任务。

## � 项目任务

本项目演示了如何使用 Web Agent 完成以下任务:

> **找到关于 Qwen 的最新技术报告(PDF),然后解读 Figure 1,描述其目的和关键发现。**

Agent 将自主完成:
1. 🔍 在网页上搜索 Qwen 技术报告
2. 📥 识别并下载最新的 PDF 文档
3. 🖼️ 从 PDF 中提取 Figure 1 图片
4. 👁️ 使用视觉模型分析图片内容
5. 📝 生成详细的解读报告

---

## 🚀 快速运行 (3步启动)

### 步骤 1: 安装依赖

```bash
# 克隆项目
git clone https://github.com/XiongBT49/WebAgent.git
cd WebAgent

# 安装 Python 依赖
pip install -r requirements.txt

# 安装浏览器驱动
playwright install chromium
```

### 步骤 2: 部署 LLM 模型

#### 🌟 方案 A: 本地部署 Ollama 

```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh
# Windows 用户访问: https://ollama.com/download

# 2. 下载模型
ollama pull qwen2.5:7b        # 文本模型 (4.7GB)
ollama pull qwen2.5vl:32b     # 视觉模型 (20GB)

# 3. 启动服务
ollama serve
# 服务运行在 http://localhost:11434
```

#### ☁️ 方案 B: 使用云端 API (无需本地资源)

**DeepSeek API** (推荐,便宜):
```bash
# 1. 获取 API Key: https://platform.deepseek.com
# 2. 配置环境变量 (见下一步)
```

**OpenAI API**:
```bash
# 1. 获取 API Key: https://platform.openai.com
# 2. 配置环境变量 (见下一步)
```

### 步骤 3: 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑配置文件
nano .env  # 或使用任何文本编辑器
```

**配置选项**:

**使用 Ollama (本地)**:
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_VISION_MODEL=qwen2.5vl:32b
```

**使用 DeepSeek (云端)**:
```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_VISION_MODEL=deepseek-vl
```

**使用 OpenAI (云端)**:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_VISION_MODEL=gpt-4-vision-preview
```

### 🎉 运行项目!

```bash
python quick_start.py
```

**输出位置**:
- 📄 下载的 PDF: `output/pdfs/`
- 🖼️ 提取的图片: `output/images/`
- 📝 执行日志: `output/logs/`

---

## 📋 环境要求

- Python 3.10+
- 8GB+ RAM
- (可选) GPU 用于本地模型推理
- (可选) 20GB+ 磁盘空间用于本地模型

---

## 🐛 常见问题

### 问题 1: "Module 'playwright' not found"

```bash
pip install playwright
playwright install chromium
```

### 问题 2: Ollama 连接失败

```bash
# 检查 Ollama 是否运行
curl http://localhost:11434/api/tags

# 如果没有响应,启动服务
ollama serve
```

### 问题 3: GPU 显存不足

```bash
# 使用更小的模型
ollama pull qwen2.5:7b     # 只用 7B 模型
```



---

## 📁 项目结构

```
WebAgent/
├── main.py              # 主 Agent 逻辑
├── tools.py             # 浏览器工具和 PDF 处理
├── config.py            # 配置加载
├── quick_start.py       # 快速启动脚本
├── requirements.txt     # Python 依赖
├── .env.example         # 配置模板
├── .env                 # 你的配置 (需要创建)
│
└── output/              # 输出目录 (自动创建)
    ├── pdfs/            # 下载的 PDF
    ├── images/          # 提取的图片
    ├── screenshots/     # 网页截图
    └── logs/            # 执行日志
```

---

## 🛠️ 核心功能

### 自动化浏览
- 搜索网页
- 点击链接
- 输入文本
- 滚动页面
- 网页截图

### PDF 处理
- 自动下载 PDF
- 提取图片
- 文本提取
- 保存输出

### 视觉理解
- 分析图表
- 解读截图
- 图片问答

---

## 🌟 特性

- ✅ 支持多种 LLM 提供商 (Ollama/DeepSeek/OpenAI)
- ✅ 视觉语言模型集成
- ✅ 自动浏览器操作
- ✅ PDF 智能处理
- ✅ 详细日志记录
- ✅ 工具调用链
- ✅ 容错重试机制

---

## � 使用示例

### 示例 1: 学术搜索

```python
from main import VLLMWebAgent

agent = VLLMWebAgent()
result = agent.run("Find the latest paper on GPT-4 from arXiv")
```

### 示例 2: 数据提取

```python
agent = VLLMWebAgent()
result = agent.run("Go to example.com and extract the table data")
```

### 示例 3: 视觉分析

```python
agent = VLLMWebAgent()
result = agent.run("Download paper.pdf and explain Figure 1")
```

