from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from agent.react_agent import ReactAgent
from agent.memory import SQLiteMemory
import json

class WindowsTroubleshootingUI:
    def __init__(self):
        self.console = Console()
        self.memory = SQLiteMemory()
        self.agent = ReactAgent(self.memory)
        self.conversation_id = "default"
    
    def display_welcome(self):
        """Display welcome message."""
        self.console.clear()
        welcome_panel = Panel(
            "# Windows OS Troubleshooting Agent\n\n"+
            "I can help you troubleshoot Windows OS issues. "+
            "Please describe your problem below.",
            title="[bold green]Welcome[/bold green]",
            border_style="green",
            expand=False
        )
        self.console.print(welcome_panel)
        self.console.print()
    
    def get_user_query(self):
        """Get user's troubleshooting query."""
        return Prompt.ask("[bold blue]Your Issue[/bold blue]")
    
    def display_plan(self, plan):
        """Display the generated troubleshooting plan."""
        self.console.print("\n[bold yellow]Generated Troubleshooting Plan:[/bold yellow]")
        
        table = Table(show_header=True, header_style="bold magenta", expand=False)
        table.add_column("Task ID", style="dim")
        table.add_column("Description")
        table.add_column("Tool")
        table.add_column("Params", style="dim")
        
        # 确保plan是字典且包含tasks键
        if not isinstance(plan, dict):
            self.console.print("[bold red]Error: Invalid plan format![/bold red]")
            return
            
        tasks = plan.get("tasks", [])
        if not tasks:
            self.console.print("[bold yellow]No tasks in the plan.[/bold yellow]")
            return
        
        self.console.print(f"Found {len(tasks)} tasks in the plan.")
        
        for task in tasks:
            # 简化params显示，避免复杂的多行JSON
            params = task.get("params", {})
            if isinstance(params, dict) and params:
                params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            else:
                params_str = "{}"
                
            # 确保所有字段都有默认值
            table.add_row(
                task.get("task_id", "N/A"),
                task.get("description", "N/A"),
                task.get("tool", "N/A"),
                params_str[:50] + "..." if len(params_str) > 50 else params_str
            )
        
        self.console.print(table)
    
    def get_human_confirmation(self):
        """Get human confirmation to execute the plan."""
        self.console.print()
        return Prompt.ask(
            "[bold yellow]Do you want to execute this plan?[/bold yellow] ",
            choices=["yes", "no"],
            default="yes"
        ) == "yes"
    
    def get_tool_execution_confirmation(self, tool_name="", params=None):
        """Get user confirmation for tool execution."""
        if tool_name and params is not None:
            self.console.print(f"\n[bold yellow]Confirm Tool Execution[/bold yellow]")
            self.console.print(f"[yellow]Tool: {tool_name}[/yellow]")
            
            if params:
                table = Table(title="Parameters", show_header=True, header_style="bold magenta")
                table.add_column("Parameter", style="dim")
                table.add_column("Value")
                
                for key, value in params.items():
                    table.add_row(key, str(value))
                
                self.console.print(table)
        
        return Confirm.ask(
            "\n[bold blue]Do you approve this tool execution?[/bold blue]",
            default=True
        )
    
    def display_tool_execution(self, task, previous_result=""):
        """Display information about the tool to be executed."""
        self.console.print("\n[bold yellow]Tool Execution Required[/bold yellow]")
        self.console.print("[yellow]The agent needs to execute a tool. Please review the details below:[/yellow]\n")
        
        # Display task information
        if task:
            self.console.print(f"[bold]Task:[/bold] {task.get('description', 'N/A')}")
            self.console.print(f"[bold]Tool:[/bold] {task.get('tool', 'N/A')}")
            
            # Display parameters in a table
            params = task.get('params', {})
            if params:
                table = Table(title="Parameters", show_header=True, header_style="bold magenta")
                table.add_column("Parameter", style="dim")
                table.add_column("Value")
                
                for key, value in params.items():
                    table.add_row(key, str(value))
                
                self.console.print(table)
            else:
                self.console.print("[dim]No parameters required[/dim]")
        
        # Display previous result if available
        if previous_result:
            self.console.print(f"\n[bold]Previous Result:[/bold]\n{previous_result}")
    
    def display_observation(self, observation):
        """Display the agent's observation of the tool result."""
        if not observation:
            return
            
        self.console.print("\n[bold purple]Agent Observation:[/bold purple]")
        
        # Display basic observation details
        obs_panel = Panel(
            f"[bold]Observation:[/bold] {observation.get('observation', 'N/A')}\n"
            f"[bold]Success:[/bold] {'Yes' if observation.get('success', False) else 'No'}\n"
            f"[bold]Details:[/bold] {observation.get('details', 'N/A')}\n"
            f"[bold]Next Steps:[/bold] {observation.get('next_steps', 'N/A')}",
            title="[bold purple]Analysis[/bold purple]",
            border_style="purple",
            expand=False
        )
        self.console.print(obs_panel)
        
        # Display LLM analysis if available
        llm_analysis = observation.get('llm_analysis')
        if llm_analysis:
            analysis_panel = Panel(
                str(llm_analysis),
                title="[bold blue]LLM Analysis[/bold blue]",
                border_style="blue",
                expand=False
            )
            self.console.print(analysis_panel)
    
    def display_plan_modifications(self, observation):
        """Display any plan modifications suggested by the agent."""
        if not observation or not observation.get('should_modify_plan', False):
            return
            
        modifications = observation.get('plan_modifications', [])
        if not modifications:
            return
            
        self.console.print("\n[bold yellow]Plan Modifications:[/bold yellow]")
        
        mod_table = Table(show_header=True, header_style="bold yellow")
        mod_table.add_column("Type")
        mod_table.add_column("Task ID")
        mod_table.add_column("Reason")
        
        for mod in modifications:
            mod_table.add_row(
                mod.get('type', 'N/A'),
                mod.get('task_id', 'N/A'),
                mod.get('reason', 'N/A')
            )
        
        self.console.print(mod_table)
    
    def display_final_result(self, result):
        """Display the final troubleshooting result."""
        self.console.print("\n" + "="*80)
        self.console.print(Markdown(result))
        self.console.print("="*80)
    
    def run(self):
        """Run the troubleshooting UI loop."""
        self.display_welcome()
        
        while True:
            user_query = self.get_user_query()
            if not user_query or user_query.lower() in ["exit", "quit", "q"]:
                self.console.print("\n[bold green]Thank you for using the Windows OS Troubleshooting Agent![/bold green]")
                break
            
            # Generate plan
            with self.console.status("[bold green]Generating plan...[/bold green]") as status:
                plan_result = self.agent.generate_plan(user_query)
            
            # Display plan and get confirmation
            self.display_plan(plan_result["plan"])
            
            if not self.get_human_confirmation():
                self.console.print("\n[bold red]Plan execution canceled.[/bold red]")
                continue
            
            # Create initial state for full execution
            import uuid
            plan_id = str(uuid.uuid4())
            
            # Mark all tasks as pending initially
            plan = plan_result["plan"]
            for task in plan.get("tasks", []):
                if "status" not in task:
                    task["status"] = "pending"
            
            initial_state = {
            "user_query": user_query,
            "plan": plan,
            "current_task": None,
            "tool_result": None,
            "conversation_history": [],
            "plan_id": plan_id,
            "pending_tool_execution": None,
            "use_llm_tool_selection": True  # Enable LLM-based tool selection by default
        }
            
            # Execute plan with a maximum iteration limit to prevent infinite loops
            config = {"recursion_limit": 200}
            current_state = initial_state
            max_iterations = 200
            iteration_count = 0
            
            while iteration_count < max_iterations:
                iteration_count += 1
                # Invoke the graph to get the next state
                current_state = self.agent.graph.invoke(current_state, config=config)
                
                # Check if there's a pending tool execution
                if current_state.get("pending_tool_execution"):
                    # Handle tool execution confirmation
                    pending_execution = current_state["pending_tool_execution"]
                    task = pending_execution.get("task", {})
                    tool_name = task.get("tool", "Unknown Tool")
                    params = task.get("params", {})
                    
                    self.display_tool_execution(task, pending_execution.get("result", ""))
                    
                    if not self.get_tool_execution_confirmation(tool_name, params):
                        self.console.print("\n[bold red]Tool execution denied. Stopping workflow.[/bold red]")
                        break
                    
                    # Execute the confirmed tool
                    result = self.agent._execute_confirmed_tool(current_state)
                    # Update current state with the result
                    current_state.update(result)
                
                # Check if we've reached the final result
                if current_state.get("tool_result") and "Troubleshooting process completed" in str(current_state.get("tool_result", "")):
                    break
            
            # Display observations from memory
            if plan_id:
                observations = self.memory.get_observations_for_plan(plan_id)
                for obs in observations:
                    # Create an observation dict similar to what the agent produces
                    observation_dict = {
                        "observation": obs["observation"],
                        "success": obs["success"],
                        "details": obs["details"],
                        "llm_analysis": obs["llm_analysis"],
                        "should_modify_plan": obs["should_modify_plan"],
                        "plan_modifications": obs["plan_modifications"]
                    }
                    
                    self.console.print(f"\n[bold purple]Observation for Task {obs['task_id']}:[/bold purple]")
                    self.display_observation(observation_dict)
                    self.display_plan_modifications(observation_dict)
            
            # Display final result
            self.display_final_result(current_state["tool_result"])
            
            # Ask if user wants to troubleshoot another issue
            self.console.print()
            cont = Prompt.ask("[bold blue]Would you like to troubleshoot another issue?[/bold blue]", choices=["yes", "no"], default="no")
            if cont != "yes":
                self.console.print("\n[bold green]Thank you for using the Windows OS Troubleshooting Agent![/bold green]")
                break

if __name__ == "__main__":
    ui = WindowsTroubleshootingUI()
    ui.agent.print_mermaid_workflow()
    ui.run()
