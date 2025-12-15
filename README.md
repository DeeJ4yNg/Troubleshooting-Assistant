# Troubleshooting Assistant

An intelligent AI-powered agent that helps diagnose and troubleshoot Windows OS issues using advanced language models and automated system diagnostics.

## Features

- **AI-Powered Troubleshooting**: Uses advanced language models to understand user problems and generate comprehensive troubleshooting plans
- **Automated Diagnostics**: Executes custom scripts and system checks to identify root causes
- **Adaptive Planning**: Dynamically modifies troubleshooting plans based on tool execution results
- **Intelligent Observation**: Analyzes tool outputs with LLM assistance to make informed decisions
- **Interactive Interface**: Rich command-line interface with clear progress indicators and results
- **Human-in-the-Loop**: Requires user confirmation before executing potentially impactful actions
- **Knowledge Base**: Built-in knowledge management system for storing and retrieving troubleshooting information

## Prerequisites

- **Operating System**: Windows 10/11 (required for Windows-specific diagnostics)
- **Python**: 3.11 or higher
- **API Access**: Inferencing service API key (OpenAI, or compatible providers)

## Quick Start

### Option 1: Using Conda (Recommended)

1. **Clone and navigate to the project:**
   ```bash
   git clone <repository-url>
   cd MyAgent_AutoFixer_SingleAgent
   ```

2. **Create and activate conda environment:**
   ```bash
   conda create -n windows-troubleshooter python=3.11
   conda activate windows-troubleshooter
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using uv (Fast Alternative)

1. **Install uv (if not already installed):**
   ```bash
   pip install uv
   ```

2. **Clone and setup project:**
   ```bash
   git clone <repository-url>
   cd MyAgent_AutoFixer_SingleAgent
   ```

3. **Create virtual environment and install dependencies:**
   ```bash
   uv venv --python python3.11
   uv pip install -r requirements.txt
   ```

4. **Activate the environment:**
   ```bash
   # On Windows (PowerShell):
   .venv\Scripts\activate
   # On Windows (Command Prompt):
   .venv\Scripts\activate.bat
   ```

### Option 3: Using Standard venv

1. **Setup project:**
   ```bash
   git clone <repository-url>
   cd MyAgent_AutoFixer_SingleAgent
   python -m venv venv
   ```

2. **Activate environment and install:**
   ```bash
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # Windows Command Prompt:
   .\venv\Scripts\activate.bat
   pip install -r requirements.txt
   ```

## Configuration

1. **Create environment file:**
   Create a new `.env` file in the project root:

2. **Configure API settings:**
   ```env
   # Required: Your LLM API key
   OPENAI_API_KEY="your-api-key-here"
   OPENAI_MODEL="gpt-4o-mini"
   OPENAI_API_BASE="https://api.openai.com/v1"
   ```

## Usage

### Basic Usage

Run the main troubleshooting agent:
```bash
python main.py
```

**Interactive Session:**
1. Describe your Windows issue when prompted
2. Review the generated troubleshooting plan
3. Confirm execution (the agent will ask for permission)
4. View the comprehensive solution report

**Example Issues to Try:**
- "Windows Update keeps failing"
- "I cannot create new .xlsx file from right-click menu"
- "I'm getting blue screen errors"

### Advanced Features

**Knowledge Management:**
```bash
python Add_Knowledge_Script.py
```
Add custom troubleshooting knowledge to the agent's knowledge base.

## How It Works

1. **Problem Analysis**: The agent tries to understand your issue and create a step-by-step troubleshooting plan
2. **Tool Selection**: Automatically selects appropriate diagnostic tools based on the problem:
   - Online search for known solutions
   - Log analysis for system errors
   - Script execution for system checks
   - Knowledge retrieval for best practices
   - Registry analysis for system settings
3. **Execution**: With your permission, executes the current planned diagnostic actions
4. **Observation & Adaptation**: Intelligently observes tool results and dynamically modifies plans based on findings
5. **Reporting**: Provides summary of findings and recommended solutions

## Available Tools

- `online_search`: Search web for solutions to common problems
- `check_system_updates`: Check for pending Windows updates
- `list_log_files`: List available system log files
- `read_event_logs`: Read Windows event logs for error patterns
- `write_ps1_file`: Create PowerShell scripts for system tasks
- `run_ps1_test`: Execute PowerShell scripts
- `add_knowledge`: Add new troubleshooting knowledge
- `search_knowledge`: Search existing knowledge base

## Project Structure

```
MyAgent_AutoDesktopFixer_SingleAgent_reconV0/
├── agent/                          # Core agent components
│   ├── react_agent.py             # Main ReAct framework implementation
│   ├── memory.py                  # SQLite-based memory system
│   ├── knowledge_manager.py       # Knowledge base management
│   ├── tools.py                   # System diagnostic tools
│   └── prompt.py                  # Prompt templates for LLM
├── utils/                         # Utility modules
│   ├── config_templates.py        # Configuration templates
│   └── logger.py                  # Logging configuration
├── main.py                        # Entry point - CLI interface
├── Add_Knowledge_Script.py       # Script for adding knowledge
├── requirements.txt              # Dependencies
└── .env                          # API configuration (user-created)
```

## Development

### Key Components

- **ReAct Agent**: Implements Reasoning-Action framework for intelligent problem-solving
- **SQLite Memory**: Persistent conversation and plan history
- **Rich UI**: Command-line interface with visual feedback
- **Tool System**: Extensible architecture for adding new diagnostic capabilities
- **Knowledge Base**: Vector-based storage for troubleshooting information

## Security Notes

⚠️ **Important Safety Considerations:**
- The agent can execute PowerShell scripts on your system
- Always review proposed plans before confirming execution
- The agent requires explicit user confirmation for all actions
- Run in a development environment first to understand behavior
- Consider the potential impact of diagnostic commands on production systems

## Troubleshooting

**Common Issues:**

1. **Import errors**: Ensure all dependencies are installed correctly
2. **API connection issues**: Verify your API key and endpoint configuration
3. **Permission errors**: Run PowerShell as Administrator for full system access
4. **Memory database errors**: Delete `agent_memory.db` and `knowledge_db.db` to reset

## Contributing

Contributions are welcome! Areas for improvement:

- **New diagnostic tools** - Adding new tools for Windows-specific issues
- **New Components** - MCP, Skills, Cost monitoring...
- **Enhanced troubleshooting workflows** - Test and investigate for better solutions
- **Better UI/UX improvements** - Frontend for better user experience
- **Better knowledge retrieval** - More accurate and relevant solutions with modern vector databases
- **Memory Management** - Better design and implementation for conversation, plan history and experience summary 

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://langchain.com) for LLM integration
- Uses [Rich](https://rich.readthedocs.io) for beautiful command-line interface
- Inspired by [ReAct](https://react-lm.github.io) AI frameworks
- Knowledge management powered by FAISS for efficient vector similarity search

---

**Ready to troubleshoot your Windows issues?** 
Start with `python main.py` and let the AI agent guide you to a solution!
