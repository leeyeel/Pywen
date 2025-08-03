# Pywen 项目结构说明

```
Pywen/
├── agents/                             # 智能体系统
|   └── qwen/                           # Qwen 智能体实现
|       ├── qwen_agent.py               # Qwen 智能体主类
|       ├── turn.py                     # Qwen 智能体回合管理
|       └── loop_detection_service.py   # Qwen 智能体循环检测服务
|   base_agent.py                       # 基础智能体类
├── config/                             # 智能体配置管理模块
│   └── config.py                       # 配置类定义，支持环境变量和配置文件加载
|   └── loader.py                       # 智能体配置加载器  
├── core/                               # 智能体核心模块
│   ├── client.py                       # 智能体级LLM 客户端，用于与大模型交互
│   ├── logger.py                       # 日志记录器
│   ├── tool_scheduler.py               # 工具调度器
│   └── tool_executor.py                # 工具执行器
|   └── trajectory_recorder.py          # 轨迹记录器
├── tools/                              # 工具生态系统
│   ├── base.py                         # 工具基类，定义所有工具的抽象接口
│   ├── bash_tool.py                    # Shell 命令执行工具
│   ├── edit_tool.py                    # 文件编辑工具
│   ├── file_tools.py                   # 文件工具（写文件、读文件）
│   ├── glob_tool.py                    # 文件 glob 工具
│   ├── grep_tool.py                    # 文件 grep 工具
│   ├── ls_tool.py                      # 文件 ls 命令工具
│   ├── memory_tool.py                  # 内存工具
│   ├── read_many_files_tool.py         # 批量读取文件工具
│   ├── 
│   ├── web_fetch_tool.py               # 网络抓取工具
│   └── web_search_tool.py              # 网络搜索工具（基于 Serper API）
├── docs/                               # 项目文档
│   ├── tools.md                        # 工具使用详细说明
│   └── project-structure.md            # 项目结构说明文档
├── trajectories/                       # 执行轨迹记录（自动生成）
│   └── trajectory_xxxxxx.json          # 单次会话的完整执行轨迹，包含 LLM 交互和工具调用
├── examples/                           # 示例代码
│   ├── code_generation/                # 代码生成示例
│   ├── project_analysis/               # 项目分析示例
│   └── tool_usage/                     # 工具使用示例
├── tests/                              # 测试套件
│   ├── test_tools/                     # 工具单元测试
│   ├── test_config/                    # 配置测试
│   └── test_integration/               # 集成测试
├── __main__.py                         # CLI 入口点，支持 `python -m pywen` 启动
├── pyproject.toml                      # 项目配置文件（依赖管理、构建配置、开发工具配置）
├── README.md                           # 项目说明（英文）
├── README_ch.md                        # 项目说明（中文）
└── pywen_config.json                   # 运行时配置文件（API 密钥、模型设置、用户偏好）
```

## 核心特点

- **模块化设计**: 基于 `BaseTool` 的插件化工具架构
- **配置管理**: 分层配置系统（环境变量 > 配置文件 > 默认值）
- **研究友好**: 完整的执行轨迹记录，便于分析和调试
- **开发体验**: 类型提示、热重载、详细文档