# Pywen

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![Python 3.12-](https://img.shields.io/badge/python-3.12-red.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Alpha](https://img.shields.io/badge/Status-Alpha-red)

[**中文版**](README_ch.md) | **English**

![Pywen Logo](./docs/Pywen.png)

**Pywen** is a Python CLI tool based on **Qwen3-Coder**, designed specifically for intelligent software engineering tasks. It provides a conversational interface that can understand natural language instructions and execute complex development workflows through an advanced agent system.

## 🎯 Project Background

Pywen is built on the [**Qwen3-Coder**](https://github.com/QwenLM/Qwen3-Coder) large language model, aiming to provide developers with an efficient and intelligent code assistant. The project is primarily adapted from [**Qwen-Code**](https://github.com/QwenLM/qwen-code), with deep optimizations for Python developers and the Qwen3-Coder model.

### Why Choose Qwen3-Coder?

- 🚀 **Code Specialization**: Qwen3-Coder excels in code generation, understanding, and debugging
- ⚡ **Efficient Inference**: Optimized model architecture providing fast responses
- 🔧 **Engineering Practicality**: Specifically trained for real-world software development scenarios

**Project Status:** The project is still under active development. We welcome your help in improving Pywen.

## Differences from Other Code Agent CLIs

Pywen is a Python-based CLI tool with excellent Python ecosystem compatibility and developer-friendliness. It provides a **transparent, modular architecture** that allows researchers and developers to easily modify, extend, and analyze, making it an ideal platform for **researching AI Agent architectures, conducting ablation studies, and developing new Agent capabilities**. This **research-friendly design** enables academia and the open-source community to more easily contribute to foundational Agent frameworks and build innovative applications, facilitating continuous breakthroughs in the rapidly evolving AI Agent field.

## ✨ Features

- 🤖 **Qwen3-Coder-Plus Powered**: Based on Alibaba Cloud's latest code-specialized large model
- 📦 **Modular**: Built on modular architecture, extensible and customizable (future support for multi-agent frameworks)
- 🛠️ **Rich Tool Ecosystem**: File editing, bash execution, sequential thinking, and more
- 📊 **Trajectory Recording**: Detailed logging of all Agent operations for debugging and analysis
- ⚙️ **Smart Configuration**: Automatic guided configuration on first run, supports environment variables
- 📈 **Session Statistics**: Real-time tracking of API calls, tool usage, and token consumption

## 🚀 Quick Start

### Installation

```bash 
pip install pywen
```

<details>
<summary>Build from source with uv (recommended)</summary>

```bash
git clone https://github.com/PAMPAS-Lab/Pywen.git
cd Pywen
uv venv
uv sync --all-extras

# linux/macos
source .venv/bin/activate

# windows
.venv\Scripts\activate
```

</details>

### First Use

Simply run the `pywen` command to start:

```bash
pywen
```

If it's your first run and there's no configuration file, Pywen will automatically start the configuration wizard:

```
██████╗ ██╗   ██╗██╗    ██╗███████╗███╗   ██║
██╔══██╗╚██╗ ██╔╝██║    ██║██╔════╝████╗  ██║
██████╔╝ ╚████╔╝ ██║ █╗ ██║█████╗  ██╔██╗ ██║
██╔═══╝   ╚██╔╝  ██║███╗██║██╔══╝  ██║╚██╗██║
██║        ██║   ╚███╔███╔╝███████╗██║ ╚████║
╚═╝        ╚═╝    ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝

Configuration file not found, starting setup wizard...

API Key: [Enter your Qwen API key]
Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
Model: qwen3-coder-plus
...

✅ Configuration saved to pywen_config.json
```

After configuration is complete, you can start using Pywen!

### Basic Usage

Once you enter the Pywen command-line interface, you can:

```bash
# File operations
> Create a Python script to calculate Fibonacci sequence
> Refactor functions in main.py to make them more efficient

# Code analysis and debugging
> Fix bugs in this project and add unit tests
> Analyze performance bottlenecks in my code

# Project management
> Create a new Flask web application with proper structure
> Add comprehensive documentation to this codebase
```

## 📖 User Guide

### Command Line Interface

#### Available Commands

```bash
# System commands
/about       show version info
/auth        change the auth method
/clear       clear the screen and conversation history
/help        for help on pywen code
/memory      Commands for interacting with memory.show
  Show       the current memory contents.add
  Add        content to the memory.refresh
  Refresh    the memory from the source.
/stats       check session stats. Usage:/stats         
/tools       list available Pywen tools 
/bug         submit a bug report
/quit        exit the cli
!            shell command                                  

# Special commands
!<command>    - Execute shell command

# Keyboard shortcuts
Ctrl+Y        - Toggle YOLO mode (auto-approve all operations - use with caution!)

# Direct input of task descriptions to execute agent
```

### YOLO Mode

**Use with caution:**
- Press `Ctrl+Y` to toggle YOLO mode
- **Default behavior**: All tool calls require user confirmation for safety
- In YOLO mode, all tool calls are automatically approved without user confirmation
- This speeds up execution but removes safety checks
- Mode status is displayed in the interface
- You can also type 'a' (always) when prompted for tool confirmation to enable YOLO mode

### Configuration Management

Pywen uses the `pywen_config.json` file for configuration:

```json
{
  "default_provider": "qwen",
  "max_steps": 20,
  "enable_lakeview": false,
  "model_providers": {
    "qwen": {
      "api_key": "your-qwen-api-key",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "model": "qwen3-coder-plus",
      "max_tokens": 4096,
      "temperature": 0.5
    }
  }
}
```

**Configuration Priority:**
1. Command line arguments (highest)
2. Configuration file values
3. Environment variables
4. Default values (lowest)

### Environment Variables

You can set API keys through environment variables:

```bash
# Qwen (required)
export QWEN_API_KEY="your-qwen-api-key"

# Tool API Keys (optional but recommended)
export SERPER_API_KEY="your-serper-api-key"  # For web search
export JINA_API_KEY="your-jina-api-key"      # For content reading

# Other supported providers
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Getting API Keys

**Serper API (Web Search):**
1. Visit [serper.dev](https://serper.dev/)
2. Sign up for a free account
3. Get your API key from the dashboard

**Jina API (Content Reading):**
1. Visit [jina.ai](https://jina.ai/)
2. Sign up for a free account
3. Get your API key from the dashboard

## 🛠️ Available Tools

Pywen provides a comprehensive toolkit for software development:

- **File Operations**: Create, edit, read, and manage files
- **Bash Execution**: Run shell commands and scripts
- **Sequential Thinking**: Structured problem-solving approach
- **Task Completion**: Mark tasks as complete with summaries
- **JSON Operations**: Parse and manipulate JSON data

For detailed information about all available tools and their capabilities, see [docs/tools.md](docs/tools.md).

## 📊 Trajectory Recording

Pywen automatically records detailed execution trajectories for debugging and analysis:

```bash
# Trajectory files are automatically saved to trajectories/ directory
trajectories/trajectory_xxxxxx.json
```

Trajectory files contain:
- **LLM Interactions**: All messages, responses, and tool calls
- **Agent Steps**: State transitions and decision points
- **Tool Usage**: Which tools were called and their results
- **Metadata**: Timestamps, token usage, and execution metrics

## 📈 Session Statistics

Monitor your usage with real-time statistics:

```bash
> /stats
```

Tracks:
- API calls and token consumption
- Tool usage patterns
- Session duration
- Model performance metrics

## 🤝 Contributing

We welcome contributions to Pywen! Here's how to get started:

1. Fork the repository
2. Set up the development environment:
   ```bash
   git clone https://github.com/your-username/Pywen.git
   cd Pywen
   uv venv
   uv sync --all-extras

   # linux/macos
   source .venv/bin/activate

   # windows
   .venv\Scripts\activate
   ```
3. Create a feature branch
4. Make your changes and add tests
5. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Use type hints appropriately
- Ensure all tests pass before submitting

## 📋 Requirements

- Python 3.9+,<3.13
- Qwen API key (recommended) or other supported LLM provider API keys
- Internet connection for API access

## 🔧 Troubleshooting

### Common Issues

**Configuration Issues:**
```bash
# Re-run configuration wizard
rm pywen_config.json
pywen
```

**API Key Issues:**
```bash
# Verify your API key is set
echo $QWEN_API_KEY

# Check configuration in Pywen
> /config
```

## 🙏 Acknowledgments

We thank:

- **Google** for the [Gemini CLI](https://github.com/google-gemini/gemini-cli) project, which provided agent execution logic and rich tool ecosystem libraries for this project
- **Alibaba Cloud Qwen Team** for providing the powerful [Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder) model and [Qwen-Code](https://github.com/QwenLM/qwen-code) reference implementation
- **ByteDance** for the [trae-agent](https://github.com/bytedance/trae-agent) project, which provided valuable foundational architecture for this project

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Pywen - Making the power of Qwen3-Coder accessible for intelligent software development!** 🚀

**PAMPAS-Lab - Dedicated to breakthroughs in large model agent frameworks, bridging AI research and applications!** 🚀

---

## 🌟Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PAMPAS-Lab/Pywen&type=Date)](https://www.star-history.com/#PAMPAS-Lab/Pywen&Date)
