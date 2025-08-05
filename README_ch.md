# Pywen

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Alpha](https://img.shields.io/badge/Status-Alpha-red)

**中文版** | [English](README.md)

**Pywen** 是一个基于 **Qwen3-Coder** 的 Python CLI 工具，专为智能软件工程任务设计。它提供对话式界面，能够理解自然语言指令并通过先进的智能体系统执行复杂的开发工作流。

## 🎯 项目背景

Pywen 核心基于[**Qwen3-Coder**](https://github.com/QwenLM/Qwen3-Coder)大语言模型，旨在为开发者提供一个高效、智能的代码助手。项目主要从[**Qwen-Code**](https://github.com/QwenLM/qwen-code)修改而来，针对 Python 开发者和 Qwen3-Coder 模型进行了深度优化。

### 为什么选择 Qwen3-Coder？

- 🚀 **代码专精**：Qwen3-Coder 在代码生成、理解和修复方面表现卓越
- ⚡ **高效推理**：优化的模型架构，提供快速响应
- 🔧 **工程实用**：专门针对实际软件开发场景训练

**项目状态：** 项目仍在积极开发中，欢迎您帮助我们改进 Pywen。

## 与其他 Code Agent CLI 的区别

Pywen 是一个基于 Python 开发的 CLI 工具，具有良好的 Python 生态兼容性和开发友好性。它提供 **透明、模块化的架构**，使研究人员和开发者可以轻松修改、扩展与分析，从而成为 **研究 AI Agent 架构、开展消融研究、开发新型 Agent 能力** 的理想平台。这种 **研究友好的设计**，让学术界与开源社区能够更便捷地为基础 Agent 框架做出贡献并构建创新应用，助力 AI Agent 快速发展领域的持续突破。


## ✨ 特性

- 🤖 **Qwen3-Coder-Plus 驱动**：基于阿里云最新的代码专用大模型
- 📦 **模块化**：基于模块化架构，可扩展和可定制(后续支持多智能体框架)
- 🛠️ **丰富的工具生态系统**：文件编辑、bash 执行、顺序思考等
- 📊 **轨迹记录**：详细记录所有 Agent 操作以供调试和分析
- ⚙️ **智能配置**：首次运行自动引导配置，支持环境变量
- 📈 **会话统计**：实时跟踪 API 调用、工具使用和Token消耗

## 🚀 快速开始

### 安装

```bash 
pip install pywen
```

<details>
<summary>使用uv从源码开始构建 (recommended)</summary>

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

### 首次使用

直接运行 `pywen` 命令即可启动：

```bash
pywen
```

如果是首次运行且没有配置文件，Pywen 会自动启动配置向导：

```
██████╗ ██╗   ██╗██╗    ██╗███████╗███╗   ██║
██╔══██╗╚██╗ ██╔╝██║    ██║██╔════╝████╗  ██║
██████╔╝ ╚████╔╝ ██║ █╗ ██║█████╗  ██╔██╗ ██║
██╔═══╝   ╚██╔╝  ██║███╗██║██╔══╝  ██║╚██╗██║
██║        ██║   ╚███╔███╔╝███████╗██║ ╚████║
╚═╝        ╚═╝    ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝

Configuration file not found, starting setup wizard...

API Key: [输入您的通义千问 API 密钥]
Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
Model: qwen3-coder-plus
...

✅ Configuration saved to pywen_config.json
```

配置完成后，您就可以开始使用 Pywen 了！

### 基本用法

进入 Pywen 命令行界面后，您可以：

```bash
# 文件操作
> 创建一个计算斐波那契数列的 Python 脚本
> 重构 main.py 中的函数，让它们更加高效

# 代码分析和调试
> 修复这个项目中的 bug 并添加单元测试
> 分析我代码中的性能瓶颈

# 项目管理
> 建立一个具有合理结构的新 Flask Web 应用
> 为这个代码库添加全面的文档
```

## 📖 使用指南

### 命令行界面

#### 可用命令

```bash
# 系统命令
/help, /?     - 显示帮助信息
/clear        - 清屏和对话历史
/status       - 显示智能体状态
/config       - 显示当前配置
/stats        - 检查会话统计（API 调用、Token等）
/quit, /exit  - 退出 Pywen

# 特殊命令
@<file>       - 在提示中包含文件内容
!<command>    - 执行 shell 命令

# 键盘快捷键
Ctrl+Y        - 切换 YOLO 模式（自动批准所有操作 - 请谨慎使用！）

# 直接输入任务描述即可执行智能体
```

### YOLO 模式

**请谨慎使用：**
- 按 `Ctrl+Y` 切换 YOLO 模式
- 在 YOLO 模式下，所有工具调用都会自动批准，无需用户确认
- 这会加快执行速度，但移除了安全检查
- 模式激活时会在界面中显示状态

### 配置管理

Pywen 使用 `pywen_config.json` 文件进行配置：

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

**配置优先级：**
1. 命令行参数（最高）
2. 配置文件值
3. 环境变量
4. 默认值（最低）

### 环境变量

您也可以通过环境变量设置 API 密钥：

```bash
# 通义千问（推荐）
export QWEN_API_KEY="your-qwen-api-key"

# 其他支持的提供商
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## 🛠️ 可用工具

Pywen 为软件开发提供了全面的工具包：

- **文件操作**：创建、编辑、读取和管理文件
- **Bash 执行**：运行 shell 命令和脚本
- **顺序思考**：结构化问题解决方法
- **任务完成**：用摘要标记任务完成
- **JSON 操作**：解析和操作 JSON 数据

有关所有可用工具及其功能的详细信息，请参阅 [docs/tools.md](docs/tools.md)。

## 📊 轨迹记录

Pywen 自动记录详细的执行轨迹以供调试和分析：

```bash
# 轨迹文件自动保存到 trajectories/ 目录
trajectories/trajectory_xxxxxx.json
```

轨迹文件包含：
- **LLM 交互**：所有消息、响应和工具调用
- **智能体步骤**：状态转换和决策点
- **工具使用**：调用了哪些工具及其结果
- **元数据**：时间戳、Token使用量和执行指标

## 📈 会话统计

通过实时统计监控您的使用情况：

```bash
> /stats
```

跟踪：
- API 调用和Token消耗
- 工具使用模式
- 会话持续时间
- 模型性能指标

## 🤝 贡献

我们欢迎为 Pywen 做出贡献！以下是开始的方法：

1. Fork 仓库
2. 设置开发环境：
   ```bash
   git clone https://github.com/your-username/Pywen.git
   cd Pywen
   uv venv
   uv sync --all-extras
   ```
3. 创建功能分支
4. 进行更改并添加测试
5. 提交拉取请求

### 开发指南

- 遵循 PEP 8 风格指南
- 为新功能添加测试
- 根据需要更新文档
- 适当使用类型提示
- 确保所有测试在提交前通过

## 📋 要求

- Python 3.9+
- 通义千问 API 密钥（推荐）或其他支持的 LLM 提供商 API 密钥
- 用于 API 访问的互联网连接

## 🔧 故障排除

### 常见问题

**配置问题：**
```bash
# 重新运行配置向导
rm pywen_config.json
pywen
```

**API 密钥问题：**
```bash
# 验证您的 API 密钥已设置
echo $QWEN_API_KEY

# 在 Pywen 中检查配置
> /config
```


## 🙏 致谢

我们感谢：

- **Google** 的[Gemini CLI](https://github.com/google-gemini/gemini-cli)项目，为本项目提供了智能体执行逻辑和丰富的工具生态库
- **阿里云通义千问团队** 提供强大的 [Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder) 模型和 [Qwen-Code](https://github.com/QwenLM/qwen-code) 参考实现
- **ByteDance** 的 [trae-agent](https://github.com/bytedance/trae-agent) 项目，为本项目提供了宝贵的基础架构

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

---

**Pywen - 让 Qwen3-Coder 的强大能力触手可及，助力智能软件开发！** 🚀

**PAMPAS-Lab - 致力于大模型智能体框架突破，为 AI 研究与应用架桥铺路！** 🚀
