#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script to showcase the fixed Windows Troubleshooting Agent with proper UI interaction.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.react_agent import ReactAgent
from agent.memory import SQLiteMemory
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from rich.panel import Panel

class FixedWindowsTroubleshootingUI:
    """Fixed UI for the Windows Troubleshooting Agent with proper state management."""
    
    def __init__(self):
        """Initialize the UI and agent."""
        self.console = Console()
        self.memory = SQLiteMemory()
        self.agent = ReactAgent(self.memory)
    
    def display_welcome(self):
        """Display the welcome message."""
        self.console.print(Panel("[bold blue]Windows Troubleshooting Agent[/bold blue]", expand=False))
        self.console.print("Welcome to the Windows Troubleshooting Agent!")
        self.console.print("I can help you diagnose and fix common Windows issues.\n")
    
    def get_user_query(self):
        """Get the user's troubleshooting query."""
        self.console.print("[bold green]What Windows issue would you like me to help you with?[/bold green]")
        query = input("> ")
        return query.strip()
    
    def display_plan(self, plan):
        """Display the troubleshooting plan."""
        self.console.print("\n[bold cyan]Proposed Troubleshooting Plan:[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Task ID", style="dim")
        table.add_column("Description")
        table.add_column("Tool")
        
        for task in plan.get("tasks", []):
            table.add_row(
                task.get("task_id", "N/A"),
                task.get("description", "N/A"),
                task.get("tool", "N/A")
            )
        
        self.console.print(table)
    
    def get_human_confirmation(self, plan):
        """Get human confirmation to proceed with the plan."""
        self.console.print("\n[bold yellow]Do you want to proceed with this plan?[/bold yellow]")
        return Confirm.ask("(y/n)", default=True)
    
    def get_tool_execution_confirmation(self, tool_name, params=None):
        """Get confirmation to execute a specific tool."""
        self.console.print(f"\n[bold yellow]About to execute tool:[/bold yellow] [green]{tool_name}[/green]")
        if params:
            self.console.print("[bold yellow]Parameters:[/bold yellow]")
            param_table = Table(show_header=True, header_style="bold magenta")
            param_table.add_column("Parameter")
            param_table.add_column("Value")
            for key, value in params.items():
                param_table.add_row(key, str(value))
            self.console.print(param_table)
        
        return Confirm.ask("[bold blue]Do you want to execute this tool?[/bold blue] (y/n)", default=True)
    
    def display_tool_execution(self, task, previous_result=""):
        """Display information about tool execution."""
        self.console.print(f"\n[yellow]Executing task:[/yellow] {task.get('description', 'Unknown task')}")
        self.console.print(f"[yellow]Tool:[/yellow] {task.get('tool', 'Unknown tool')}")
        
        # Show parameters if available
        params = task.get('params', {})
        if params:
            self.console.print("[yellow]Parameters:[/yellow]")
            param_table = Table(show_header=True, header_style="bold magenta")
            param_table.add_column("Parameter")
            param_table.add_column("Value")
            for key, value in params.items():
                param_table.add_row(key, str(value))
            self.console.print(param_table)
        
        # Show previous result if available
        if previous_result:
            self.console.print(f"[yellow]Previous result:[/yellow] {previous_result}")
    
    def display_observation(self, observation):
        """Display the observation of tool execution."""
        if observation:
            success = observation.get("success", False)
            status_color = "green" if success else "red"
            status_text = "Success" if success else "Failed"
            
            self.console.print(f"\n[{status_color}]Task Status: {status_text}[/{status_color}]")
            self.console.print(f"[cyan]Observation:[/cyan] {observation.get('observation', 'No observation')}")
            self.console.print(f"[cyan]Details:[/cyan] {observation.get('details', 'No details')}")
    
    def display_plan_modifications(self, modifications):
        """Display any plan modifications."""
        if modifications:
            self.console.print("\n[bold purple]Plan Modifications:[/bold purple]")
            for mod in modifications:
                mod_type = mod.get("type", "unknown")
                reason = mod.get("reason", "No reason provided")
                self.console.print(f"  [{mod_type.upper()}] {reason}")
    
    def display_final_result(self, final_state):
        """Display the final result."""
        self.console.print("\n[bold green]Troubleshooting Complete![/bold green]")
        result = final_state.get("final_result", "No result available")
        self.console.print(Panel(result, title="Final Result", border_style="green"))
    
    def run(self):
        """Run the troubleshooting agent with proper UI interaction."""
        # Display welcome message
        self.display_welcome()
        
        # Get user query
        user_query = self.get_user_query()
        if not user_query:
            self.console.print("[red]No query provided. Exiting.[/red]")
            return
        
        # Initialize the agent state
        initial_state = {
            "user_query": user_query,
            "plan": None,
            "current_task": None,
            "tool_result": None,
            "conversation_history": [],
            "plan_id": str(hash(user_query)),  # Simple plan ID generation
            "observation": None,
            "pending_tool_execution": None
        }
        
        # Configuration for the graph
        config = {"recursion_limit": 50}
        
        try:
            # Start the workflow
            current_state = initial_state
            
            # Invoke the graph
            current_state = self.agent.graph.invoke(current_state, config=config)
            
            # Display final result
            self.display_final_result(current_state)
            
        except Exception as e:
            self.console.print(f"\n[red]Error during troubleshooting: {str(e)}[/red]")
            self.console.print_exception()
    
    def run_interactive_demo(self):
        """Run an interactive demo showcasing the fixed functionality."""
        self.console.print(Panel("[bold]Interactive Demo of Fixed Windows Troubleshooting Agent[/bold]", expand=False))
        self.console.print("This demo showcases the fixed functionality of the agent:")
        self.console.print("1. Proper state management with Annotated types")
        self.console.print("2. Correct tool execution flow")
        self.console.print("3. UI interaction with confirmation prompts")
        self.console.print("4. Proper error handling\n")
        
        if Confirm.ask("[bold blue]Would you like to run the demo?[/bold blue]", default=True):
            self.run()
        else:
            self.console.print("[yellow]Demo cancelled.[/yellow]")

def main():
    """Main function to run the demo."""
    ui = FixedWindowsTroubleshootingUI()
    ui.run_interactive_demo()

if __name__ == "__main__":
    main()