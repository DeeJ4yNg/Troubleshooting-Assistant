#!/usr/bin/env python3
"""
Windows OS Troubleshooting Agent - Rich UI Demo
Demonstrates the enhanced observation and plan modification capabilities with a beautiful UI.
"""

import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import box
from agent.react_agent import ReactAgent
from agent.memory import SQLiteMemory

# Define emoji aliases for compatibility
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

def main():
    console = Console()
    
    # Header
    console.print(Panel("[bold blue]Windows OS Troubleshooting Agent[/bold blue]\n[italic]Enhanced Observation & Adaptive Planning Demo[/italic]", expand=False))
    
    # Initialize the agent
    console.print("[bold green][Wrench] Initializing Agent...[/bold green]")
    memory = SQLiteMemory()
    agent = ReactAgent(memory)
    
    # Example query
    query = "My computer is running extremely slow and sometimes freezes"
    
    console.print(Panel(f"[bold]User Query:[/bold] {query}", border_style="blue"))
    
    # Generate initial plan
    console.print("\n[bold cyan]1. Generating Initial Troubleshooting Plan[/bold cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing problem and generating plan...", total=None)
        result = agent.generate_plan(query)
        plan = result["plan"]
        progress.update(task, completed=True)
    
    # Display initial plan
    table = Table(title="[Clipboard] Initial Troubleshooting Plan", box=box.ROUNDED)
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Task Description", style="white")
    table.add_column("Tool", style="green")
    
    for i, task in enumerate(plan['tasks'], 1):
        table.add_row(str(i), task['description'], task['tool'])
    
    console.print(table)
    console.print(f"[bold green][Checkmark][/bold green] Generated plan with [bold]{len(plan['tasks'])}[/bold] tasks\n")
    
    # Simulate executing the first task (online search)
    console.print("[bold cyan]2. Simulating Tool Execution[/bold cyan]")
    first_task = plan['tasks'][0]
    console.print(Panel(f"[bold]Executing:[/bold] {first_task['description']}", border_style="yellow"))
    
    # Mock tool result (as if from online_search tool)
    mock_tool_result = [
        {
            "title": "How to Fix Slow Computer Performance",
            "url": "https://example.com/slow-computer-fixes",
            "content": "Common causes of slow computer performance include insufficient RAM, too many startup programs, malware infections, and fragmented hard drives. Solutions include upgrading RAM, disabling unnecessary startup programs, running antivirus scans, and defragmenting the hard drive."
        },
        {
            "title": "Windows 10 Slow Performance Troubleshooting",
            "url": "https://example.com/windows-10-performance",
            "content": "For Windows 10 slow performance, check Task Manager for resource-heavy processes, run Disk Cleanup, update drivers, and consider upgrading to an SSD if using a traditional hard drive."
        }
    ]
    
    # Pretty print the tool result
    result_table = Table(box=box.SIMPLE)
    result_table.add_column("Title", style="bold")
    result_table.add_column("Content")
    
    for item in mock_tool_result:
        result_table.add_row(item["title"], item["content"][:100] + "...")
    
    console.print("[bold]Tool Result:[/bold]")
    console.print(result_table)
    
    # Create a mock state as if after tool execution
    mock_state = {
        "user_query": query,
        "plan": plan,
        "current_task": first_task,
        "tool_result": mock_tool_result,
        "conversation_history": [],
        "plan_id": result["plan_id"]
    }
    
    # Observe the result (this is where our enhancement kicks in)
    console.print("\n[bold cyan]3. Intelligent Observation with LLM Analysis[/bold cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing tool result and generating observation...", total=None)
        observed_state = agent._observe_result(mock_state)
        observation = observed_state.get("observation", {})
        progress.update(task, completed=True)
    
    # Display observation
    obs_panel = Panel(
        f"""[bold]Status:[/bold] {'[Checkmark] Success' if observation.get('success') else '[X] Failure'}
[bold]Details:[/bold] {observation.get('details')}
[bold]Next Steps:[/bold] {observation.get('next_steps')}
[bold]Plan Modification Needed:[/bold] {'Yes' if observation.get('should_modify_plan') else 'No'}""",
        title="[MagnifyingGlassTilted] Observation Results",
        border_style="green" if observation.get('success') else "red"
    )
    console.print(obs_panel)
    
    # Show LLM analysis if available
    if observation.get('llm_analysis'):
        analysis_panel = Panel(
            f"[italic]{observation['llm_analysis']}[/italic]",
            title="[Robot] LLM Analysis",
            border_style="purple"
        )
        console.print(analysis_panel)
    
    # Show plan modifications suggested by LLM
    if observation.get('plan_modifications'):
        mod_table = Table(title="[HammerAndWrench] Suggested Plan Modifications", box=box.ROUNDED)
        mod_table.add_column("Type", style="cyan")
        mod_table.add_column("Reason", style="white")
        mod_table.add_column("Details", style="yellow")
        
        for mod in observation['plan_modifications']:
            details = ""
            if mod['type'] == 'add' and 'new_task' in mod:
                details = mod['new_task'].get('description', 'N/A')
            elif mod['type'] == 'remove' or mod['type'] == 'modify':
                details = f"Task ID: {mod.get('task_id', 'N/A')}"
            
            mod_table.add_row(mod['type'].capitalize(), mod['reason'], details)
        
        console.print(mod_table)
    
    # Apply plan modifications
    console.print("\n[bold cyan]4. Dynamic Plan Modification[/bold cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Applying plan modifications...", total=None)
        modified_state = agent._modify_plan(observed_state)
        progress.update(task, completed=True)
    
    # Show the updated plan
    updated_plan = modified_state["plan"]
    console.print(f"[bold green][Checkmark][/bold green] Plan updated successfully!\n")
    
    updated_table = Table(title="[Clipboard] Updated Troubleshooting Plan", box=box.ROUNDED)
    updated_table.add_column("Step", style="cyan", no_wrap=True)
    updated_table.add_column("Status", style="white")
    updated_table.add_column("Task Description", style="white")
    updated_table.add_column("Tool", style="green")
    
    for i, task in enumerate(updated_plan['tasks'], 1):
        status = task.get('status', 'unknown')
        status_icon = "[Checkmark]" if status == "completed" else "[Hourglass]" if status == "pending" else "[QuestionMark]"
        updated_table.add_row(str(i), f"{status_icon} {status}", task['description'], task['tool'])
    
    console.print(updated_table)
    
    # Final summary
    console.print("\n[bold green][PartyPopper] Demonstration Complete![/bold green]")
    summary_panel = Panel("""[bold]The Windows OS Troubleshooting Agent successfully demonstrated:[/bold]

[Sparkles] [bold blue]Intelligent Observation[/bold blue]
   [Bullet] Analyzed tool results with LLM assistance
   [Bullet] Provided structured observations with success/failure detection

[Zap] [bold magenta]Adaptive Planning[/bold magenta]
   [Bullet] Dynamically modified plans based on findings
   [Bullet] Added new tasks based on discovered information

[ClockwiseVerticalArrows] [bold yellow]Context Awareness[/bold yellow]
   [Bullet] Maintained execution context throughout the process
   [Bullet] Preserved conversation history for transparency

[Shield] [bold green]Robust Implementation[/bold green]
   [Bullet] Handled various result formats gracefully
   [Bullet] Provided clear user feedback at each step""",
        title="[Trophy] Key Capabilities",
        border_style="green"
    )
    console.print(summary_panel)

if __name__ == "__main__":
    main()