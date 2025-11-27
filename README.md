# Windows OS Troubleshooting Agent

An intelligent agent that helps diagnose and troubleshoot Windows OS issues using AI-powered planning and execution.

## Features

- **AI-Powered Troubleshooting**: Uses advanced language models to understand user problems and generate comprehensive troubleshooting plans
- **Automated Diagnostics**: Executes PowerShell scripts and system checks to identify root causes
- **Adaptive Planning**: Dynamically modifies troubleshooting plans based on tool execution results
- **Intelligent Observation**: Analyzes tool outputs with LLM assistance to make informed decisions
- **Interactive Interface**: Rich command-line interface with clear progress indicators and results
- **Human-in-the-Loop**: Requires user confirmation before executing potentially impactful actions
- **Persistent Memory**: Remembers past interactions for improved troubleshooting over time

## Prerequisites

- Python 3.9+
- Windows OS (tested on Windows 10/11)
- API key for an LLM service (OpenAI, SiliconFlow, etc.)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MyAgent_AutoDesktopFixer_SingleAgent
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your API key:
   Create a `.env` file in the project root with your API credentials:
   ```env
   OPENAI_API_KEY="your-api-key-here"
   OPENAI_MODEL="Qwen/Qwen2.5-7B-Instruct"  # or your preferred model
   OPENAI_API_BASE="https://api.siliconflow.cn/v1"  # or your API endpoint
   ```

## Usage

Run the agent with:
```bash
python main.py
```

Then describe your Windows issue when prompted. The agent will:
1. Generate a detailed troubleshooting plan
2. Ask for your confirmation before proceeding
3. Execute the plan using various system tools
4. Present a comprehensive solution report

### Rich UI Demo

To see the enhanced capabilities with a beautiful visual interface, run the demo:
```bash
python demo_rich_ui.py
```

This demo showcases:
- Intelligent observation of tool results with LLM analysis
- Dynamic plan modification based on findings
- Beautiful Rich UI with panels, tables, and progress indicators

## How It Works

1. **Problem Analysis**: The agent uses an LLM to understand your issue and create a step-by-step troubleshooting plan
2. **Tool Selection**: Based on the problem, it selects appropriate tools like:
   - Online search for known solutions
   - Event log analysis for system errors
   - PowerShell script execution for system checks
   - Knowledge retrieval for best practices
3. **Execution**: With your permission, it executes the planned actions
4. **Observation & Adaptation**: The agent intelligently observes tool results and can dynamically modify its plan based on findings
5. **Reporting**: Provides a clear summary of findings and recommended solutions

## Example Tools

- `online_search`: Searches the web for solutions to common problems
- `check_system_updates`: Checks for pending Windows updates
- `list_log_files`: Lists available system log files
- `read_event_logs`: Reads Windows event logs for error patterns
- `write_ps1_file`: Creates PowerShell scripts for system tasks
- `run_ps1_test`: Executes PowerShell scripts safely
- `report_result`: Compiles findings into a readable report

## Development

### Project Structure

```
├── agent/
│   ├── react_agent.py    # Main agent logic using ReAct framework
│   ├── memory.py         # SQLite-based memory system
│   └── tools.py          # System tools implementation
├── main.py              # Entry point with CLI interface
├── requirements.txt     # Python dependencies
└── .env                 # API configuration (not included in repo)
```

### Key Components

1. **ReAct Agent**: Implements the Reasoning-action framework for problem-solving
2. **SQLite Memory**: Stores conversation history and learned patterns
3. **Rich UI**: Provides an intuitive command-line interface with visual feedback
4. **Tool System**: Extensible collection of system utilities for diagnostics

## Security Notes

- The agent can execute PowerShell scripts on your system
- Always review the proposed plan before confirming execution
- The agent only executes actions with your explicit confirmation

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for:

- New diagnostic tools
- Improved troubleshooting workflows
- Better UI/UX enhancements
- Additional LLM integrations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with LangChain for LLM integration
- Uses Rich for the command-line interface
- Inspired by ReAct (Reasoning-action) AI frameworks