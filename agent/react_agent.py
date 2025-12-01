from typing import Dict, Any, List, Optional, Sequence, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from agent.memory import SQLiteMemory
from agent.tools import tools
import uuid
import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.channels.base import BaseChannel
from langgraph.channels.last_value import LastValue
from langgraph.channels.ephemeral_value import EphemeralValue
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# Load environment variables
load_dotenv()

# Define the state schema for the agent
class AgentState(TypedDict):
    """State definition for the LangGraph agent."""
    user_query: Annotated[str, LastValue(str)]
    plan: Annotated[Optional[Dict], LastValue(Optional[Dict])]
    current_task: Annotated[Optional[Dict], LastValue(Optional[Dict])]
    tool_result: Annotated[Optional[Any], LastValue(Optional[Any])]
    conversation_history: Annotated[List[Dict], LastValue(List[Dict])]
    plan_id: Annotated[Optional[str], LastValue(Optional[str])]
    observation: Annotated[Optional[Dict], LastValue(Optional[Dict])]
    pending_tool_execution: Annotated[Optional[Dict], LastValue(Optional[Dict])]

class ReactAgent:
    def __init__(self, memory: SQLiteMemory):
        self.memory = memory
        # Initialize OpenAI LLM with API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        api_base = os.getenv("OPENAI_API_BASE")
        self.console = Console()
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.llm = ChatOpenAI(
            base_url=api_base,
            model=openai_model,
            openai_api_key=openai_api_key,
            temperature=0.7
        )
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow graph."""
        # Create the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("create_plan", self._create_plan)
        workflow.add_node("get_human_confirmation", self._get_human_confirmation)
        workflow.add_node("execute_task", self._execute_task)
        workflow.add_node("execute_confirmed_tool", self._execute_confirmed_tool)
        workflow.add_node("observe_result", self._observe_result)
        workflow.add_node("modify_plan", self._modify_plan)
        workflow.add_node("check_completion", self._check_completion)
        workflow.add_node("report_result", self._report_result)
        
        # Add edges
        workflow.set_entry_point("create_plan")
        workflow.add_edge("create_plan", "get_human_confirmation")
        workflow.add_edge("execute_confirmed_tool", "observe_result")
        workflow.add_edge("observe_result", "modify_plan")
        workflow.add_edge("modify_plan", "check_completion")
        
        # Add conditional edges for tool execution flow
        workflow.add_conditional_edges(
            "execute_task",
            self._route_after_task_execution,
            {
                "awaiting_confirmation": "get_human_confirmation",
                "continue": "observe_result"
            }
        )
        
        # Add conditional edges for tool confirmation flow
        workflow.add_conditional_edges(
            "get_human_confirmation",
            self._route_after_human_confirmation,
            {
                "execute_tool": "execute_confirmed_tool",
                "cancel_exe": END
            }
        )
        
        # Add conditional edges for plan completion
        workflow.add_conditional_edges(
            "check_completion",
            self._should_continue,
            {
                "continue": "execute_task",
                "complete": "report_result"
            }
        )
        
        workflow.add_edge("report_result", END)
        
        return workflow.compile()
    
    def _create_plan(self, state: AgentState) -> Dict:
        """Create a troubleshooting plan based on the user query."""
        prompt = ChatPromptTemplate.from_template("""
        You are a Windows OS troubleshooting expert. Create a detailed step-by-step plan to troubleshoot the following issue:
        
        {user_query}
        
        The plan should include multiple specific tasks that can be executed using the following tools:
        - list_log_files: List log files in a directory (Example params: {{"path": "C:/Windows/Logs"}})
        - read_error_logs: Read error messages from log files (Example params: {{"file_path": "C:/Windows/Logs/System.evtx", "max_lines": 100}})
        - write_ps1_file: Create PowerShell scripts (Example params: {{"file_path": "C:/temp/fix_settings.ps1", "content": "Get-AppxPackage Microsoft.Windows.SettingsApp | Reset-AppxPackage"}})
        - run_ps1_test: Run PowerShell scripts (Example params: {{"file_path": "C:/temp/fix_settings.ps1"}})
        - online_search: Search the internet (Example params: {{"query": "Windows 11 Settings app not opening fix", "max_results": 3}})
        - knowledge_retrieval: Retrieve knowledge from database (Example params: {{"query": "Windows Settings app repair methods", "limit": 5}})
        
        Create a comprehensive plan with at least 3-5 tasks that thoroughly address the issue.
        Each task should have:
        1. A unique task_id (e.g., "task_1", "task_2")
        2. A clear description of what the task does
        3. The appropriate tool to use
        4. Proper parameters for the tool
        
        Format the plan as a JSON object with the following structure:
        {{
            "tasks": [
                {{
                    "task_id": "task_1",
                    "description": "Description of the task",
                    "tool": "online_search",
                    "params": {{
                        "query": "{user_query}"
                    }}
                }}
            ]
        }}
        
        Make sure to escape all curly braces in the JSON structure. The user's query is: {user_query}
        """)
        
        if state.get("plan"):
            return state

        chain = prompt | self.llm | StrOutputParser()
        plan_str = chain.invoke({"user_query": state["user_query"]})
        print(f"Generated plan: {plan_str}")  # Debug print
        
        # Parse the plan (simplified for now)
        import json
        try:
            plan = json.loads(plan_str)
        except json.JSONDecodeError:
            # Fallback plan if LLM doesn't return valid JSON
            plan = {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "description": "Search for information about the issue",
                        "tool": "online_search",
                        "params": {"query": state["user_query"]}
                    }
                ]
            }
        
        # Save plan to memory
        self.memory.save_plan(state["plan_id"], state["user_query"], plan)
        
        return {
            "user_query": state["user_query"],
            "plan": plan,
            "conversation_history": state["conversation_history"],
            "plan_id": state["plan_id"]
        }
    
    def _get_human_confirmation(self, state: AgentState) -> Dict:
        """Get human confirmation before executing the plan or a tool."""
        # This method will be called by the workflow, but the actual UI interaction
        # will happen in the main loop. For now, we'll just pass through the state.
        print("Getting human confirmation...")
        
        # If there's already pending_tool_execution in state, use it directly
        pending_execution = state.get("pending_tool_execution")
        if pending_execution:
            return state
        
        # If no pending execution, find the first non-completed task
        plan = state.get("plan", {})
        tasks = plan.get("tasks", [])
        
        current_task = None
        for task in tasks:
            if task.get("status") != "completed":
                current_task = task
                break
        
        if not current_task:
            return state
        
        tool_name = current_task.get("tool")
        params = current_task.get("params", {})
        
        # Get the tool instance
        tool = tools.get(tool_name)
        if not tool:
            tool_result = f"Error: Tool '{tool_name}' not found"
            return {
                "user_query": state["user_query"],
                "plan": state["plan"],
                "current_task": current_task,
                "tool_result": tool_result,
                "conversation_history": state["conversation_history"],
                "plan_id": state["plan_id"]
            }
        else:
            # Store tool information in state for UI confirmation
            pending_execution = {
                "tool_name": tool_name,
                "params": params,
                "task": current_task
            }
            
            # Return early to allow UI to confirm execution
            return {
                "user_query": state["user_query"],
                "plan": state["plan"],
                "current_task": current_task,
                "tool_result": None,
                "conversation_history": state["conversation_history"],
                "plan_id": state["plan_id"],
                "pending_tool_execution": pending_execution
            }
    
    def _execute_task(self, state: AgentState) -> Dict:
        """Execute the current task from the plan."""
        # Get the current task from the plan
        plan = state.get("plan", {})
        tasks = plan.get("tasks", [])

        # Find the first pending task
        current_task = None
        for task in tasks:
            if task.get("status") != "completed":
                current_task = task
                print(f"Executing task: {current_task}")  # Debug print
                break
        
        if not current_task:
            # No pending tasks found
            return {
                "user_query": state["user_query"],
                "plan": state["plan"],
                "current_task": None,
                "tool_result": "No pending tasks found in the plan.",
                "conversation_history": state["conversation_history"],
                "plan_id": state["plan_id"]
            }
        
        # Check if the task requires a tool
        tool_name = current_task.get("tool")
        if tool_name:
            # Task requires a tool - prepare for execution or confirmation
            params = current_task.get("params", {})
            
            # Get the tool instance
            tool = tools.get(tool_name)
            if not tool:
                # Tool not found
                return {
                    "user_query": state["user_query"],
                    "plan": state["plan"],
                    "current_task": current_task,
                    "tool_result": f"Error: Tool '{tool_name}' not found",
                    "conversation_history": state["conversation_history"],
                    "plan_id": state["plan_id"]
                }
            else:
                # For certain tools, we might need human confirmation
                # For now, we'll set up pending execution for UI confirmation
                pending_execution = {
                    "tool_name": tool_name,
                    "params": params,
                    "task": current_task
                }
                
                return {
                    "user_query": state["user_query"],
                    "plan": state["plan"],
                    "current_task": current_task,
                    "conversation_history": state["conversation_history"],
                    "plan_id": state["plan_id"],
                    "pending_tool_execution": pending_execution
                }
        else:
            # Task doesn't require a tool - mark as completed
            current_task["status"] = "completed"
            return {
                "user_query": state["user_query"],
                "plan": state["plan"],
                "current_task": current_task,
                "tool_result": f"Task completed: {current_task.get('description')}",
                "conversation_history": state["conversation_history"],
                "plan_id": state["plan_id"]
            }
    
    def _execute_confirmed_tool(self, state: AgentState) -> Dict:
        """Execute a tool that has been confirmed by the user."""
        pending_execution = state.get("pending_tool_execution")
        if not pending_execution:
            return state
            
        tool_name = pending_execution["tool_name"]
        params = pending_execution["params"]
        confirm = Prompt.ask(f"[bold yellow]Are you sure you want to execute tool: {tool_name} with params: {params}? (yes/no)[/bold yellow]")
        if confirm.lower() != "yes":
            return state
        # Get the tool instance
        tool = tools.get(tool_name)
        if not tool:
            tool_result = f"Error: Tool '{tool_name}' not found"
        else:
            # Execute the tool
            self.console.print(f"Executing tool: {tool_name} with params: {params}")  # Debug print
            try:
                tool_result = tool(**params)
            except Exception as e:
                tool_result = f"Error executing tool '{tool_name}': {str(e)}"
        
        # Clear pending execution and return result
        return {
            "user_query": state["user_query"],
            "plan": state["plan"],
            "current_task": pending_execution.get("task"),
            "conversation_history": state["conversation_history"],
            "plan_id": state["plan_id"],
            "pending_tool_execution": None,  # Clear pending execution
            "tool_result": tool_result  # Add tool execution result
        }
    
    def _observe_result(self, state: AgentState) -> Dict:
        """Observe and analyze the tool execution result using LLM intelligence."""
        tool_result = state.get("tool_result")
        current_task = state.get("current_task") or {}
        task_description = current_task.get("description", "Unknown task")
        observation = {
            "success": False,
            #"details": "",
            #"next_steps": "",
            "plan_modifications": []
            #"observation": ""
        }
        
        # Use LLM to analyze the result and suggest plan modifications
        try:
            analysis_prompt = ChatPromptTemplate.from_template("""
            You are a Windows OS troubleshooting expert. Analyze the following tool execution result and suggest 
            any necessary modifications to the troubleshooting plan.
            
            IMPORTANT: Avoid creating duplicate tasks that were already executed in the conversation history. 
            DO NOT fall into infinite loops by repeatedly executing similar tools or asking similar questions.
            Focus on making progress toward resolving the issue rather than repeating previous steps.
            
            Original user query: {user_query}
            
            Conversation history:
            {conversation_history}
            
            Executed task: {task_description}
            Tool result: {tool_result}
            
            Current plan tasks:
            {plan_tasks}
            
            Based on the tool result and conversation history, determine if:
            1. The plan should be modified (add new tasks, remove unnecessary or failed tasks, change task order)
            2. Additional investigation is needed
            3. The troubleshooting approach should be adjusted
            4. Only use the tools that are available in the tools available list .
            
            Before adding new tasks, verify they don't duplicate previous actions from conversation history.
            If you see a pattern of repeated tool executions without progress, change the approach.

            Tools available:
            - list_log_files: List log files in a directory (Example params: {{"path": "C:/Windows/Logs"}})
            - read_error_logs: Read error messages from log files (Example params: {{"file_path": "C:/Windows/Logs/System.evtx", "max_lines": 100}})
            - write_ps1_file: Create PowerShell scripts (Example params: {{"file_path": "C:/temp/fix_settings.ps1", "content": "Get-AppxPackage Microsoft.Windows.SettingsApp | Reset-AppxPackage"}})
            - run_ps1_test: Run PowerShell scripts (Example params: {{"file_path": "C:/temp/fix_settings.ps1"}})
            - online_search: Search the internet (Example params: {{"query": "Windows 11 Settings app not opening fix", "max_results": 3}})
            - knowledge_retrieval: Retrieve knowledge from database (Example params: {{"query": "Windows Settings app repair methods", "limit": 5}})
            
            Respond with a JSON object containing:
            {{
                "success": true/false,
                "analysis": "Brief analysis of the result",
                "should_modify_plan": true/false,
                "modifications": [
                    {{
                        "type": "add/remove/modify",
                        "task_id": "task_id_if_applicable",
                        "reason": "reason_for_modification",
                        "new_task": {{}} // Only if type is "add"
                    }}
                ]
            }}
            
            Keep modifications minimal and focused. If no changes are needed, set should_modify_plan to false.
            Always prioritize making forward progress and avoiding repetition.

            REMEMBER: NO DUPLICATE TASKS IN THE PLAN!
            """)
            
            plan_tasks_info = "\n".join([
                f"- {task.get('task_id', 'N/A')}: {task.get('description', 'N/A')} (Status: {task.get('status', 'pending')})"
                for task in state.get("plan", {}).get("tasks", [])
            ])
            
            chain = analysis_prompt | self.llm | StrOutputParser()
            
            # Format conversation history for the prompt
            conversation_history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in state.get("conversation_history", [])[-10:]  # Include last 10 messages to keep context manageable
            ])
            
            with self.console.status("[bold green]Analyzing tool result...[/bold green]") as status:
                analysis_result = chain.invoke({
                "user_query": state["user_query"],
                "conversation_history": conversation_history_text,
                "task_description": task_description,
                "tool_result": str(tool_result),
                "plan_tasks": plan_tasks_info
                })
            
            # Try to parse the LLM response as JSON
            import json
            try:
                analysis_json = json.loads(analysis_result)
                self.console.print(analysis_json)
                observation["success"] = analysis_json.get("success", False)
                observation["llm_analysis"] = analysis_json.get("analysis", "")
                observation["should_modify_plan"] = analysis_json.get("should_modify_plan", False)
                observation["plan_modifications"] = analysis_json.get("modifications", [])
            except json.JSONDecodeError:
                # If JSON parsing fails, treat as plain text analysis
                observation["llm_analysis"] = analysis_result
                observation["should_modify_plan"] = False
                observation["plan_modifications"] = []
                
        except Exception as e:
            # If LLM analysis fails, continue with basic observation
            observation["llm_analysis"] = f"LLM analysis failed: {str(e)}"
            observation["should_modify_plan"] = False
            observation["plan_modifications"] = []
        
        # Store observation in memory
        if state.get("plan_id") and state.get("current_task"):
            self.memory.save_observation(
                plan_id=state["plan_id"],
                task_id=state["current_task"]["task_id"],
                observation="",
                success=observation["success"],
                details={},
                llm_analysis={
                    "analysis": observation.get("llm_analysis", ""),
                    "should_modify_plan": observation.get("should_modify_plan", False),
                    "modifications": observation.get("plan_modifications", [])
                },
                should_modify_plan=observation.get("should_modify_plan", False),
                plan_modifications=observation.get("plan_modifications", [])
            )
        
        # Add the observation to the conversation history
        conversation_history = state.get("conversation_history", []).copy()
        conversation_history.append({
            "role": "system",
            "content": f"Tool_Result: {tool_result}. modifications: {observation.get('plan_modifications', '')}. LLM Analysis: {observation.get('llm_analysis', '')}"
        })
        
        return {
            "user_query": state["user_query"],
            "plan": state["plan"],
            "current_task": state["current_task"],
            "tool_result": tool_result,
            "conversation_history": conversation_history,
            "plan_id": state["plan_id"],
            "observation": observation
        }
    

    def _modify_plan(self, state: AgentState) -> Dict:
        """Modify the plan based on the tool result and LLM analysis."""
        # Mark current task as completed
        if state.get("current_task"):
            for task in state["plan"].get("tasks", []):
                if task["task_id"] == state["current_task"]["task_id"]:
                    task["status"] = "completed"
                    break
        
        # Fall back to existing observation-based modifications if no tool result or LLM fails
        if state.get("observation", {}).get("should_modify_plan", False):
            modifications = state.get("observation", {}).get("plan_modifications", [])
            plan_tasks = state["plan"].get("tasks", [])
            
            for modification in modifications:
                mod_type = modification.get("type")
                task_id = modification.get("task_id")
                reason = modification.get("reason")
                
                if mod_type == "add" and "new_task" in modification:
                    # Add new task
                    new_task = modification["new_task"]
                    # Ensure the new task has all required fields
                    if "task_id" not in new_task:
                        # Generate a new task ID
                        import uuid
                        new_task["task_id"] = f"task_{uuid.uuid4().hex[:8]}"
                    if "status" not in new_task:
                        new_task["status"] = "pending"
                    if "description" not in new_task:
                        new_task["description"] = "New task added by LLM observation"
                    if "tool" not in new_task:
                        new_task["tool"] = "knowledge_retrieval"  # Default tool
                    if "params" not in new_task:
                        new_task["params"] = {}
                    
                    # Ensure knowledge_retrieval tasks have a query parameter
                    if new_task["tool"] == "knowledge_retrieval" and "query" not in new_task["params"]:
                        new_task["params"]["query"] = state.get("user_query", "")
                    
                    plan_tasks.append(new_task)
                    self.console.print(f"[bold yellow]Added new task: {new_task.get('description', 'N/A')} - Reason: {reason}[/bold yellow]")
                    
                elif mod_type == "remove" and task_id:
                    # Remove task
                    plan_tasks = [task for task in plan_tasks if task.get("task_id") != task_id]
                    self.console.print(f"[bold yellow]Removed task {task_id} - Reason: {reason}[/bold yellow]")
                    
                elif mod_type == "modify" and task_id:
                    # Modify existing task
                    for task in plan_tasks:
                        if task.get("task_id") == task_id:
                            # Don't reset task status to pending to avoid re-execution
                            # Only update task properties without changing status
                            # Ensure knowledge_retrieval tasks have a query parameter
                            if task.get("tool") == "knowledge_retrieval" and "params" in task:
                                if "query" not in task["params"]:
                                    task["params"]["query"] = state.get("user_query", "")
                            self.console.print(f"[bold yellow]Modified task {task_id} - Reason: {reason}[/bold yellow]")
                            break
            
            # Update the plan with modified tasks
            state["plan"]["tasks"] = plan_tasks
        
        # Save updated plan to memory
        self.memory.save_plan(state["plan_id"], state["user_query"], state["plan"])
        
        # Show current plan as a table.
        self.console.print("\n[bold yellow]Current Troubleshooting Plan:[/bold yellow]")
        table = Table(show_header=True, header_style="bold magenta", expand=False)
        table.add_column("Task ID", style="dim")
        table.add_column("Description")
        table.add_column("Tool")
        table.add_column("Params", style="dim")
        table.add_column("Status")
        for task in state["plan"]["tasks"]:
            table.add_row(
                task["task_id"],
                task.get("description", "N/A"),
                task.get("tool", "N/A"),
                str(task.get("params", {})),
                task.get("status", "N/A")
            )
        self.console.print(table)

        return {
            "user_query": state["user_query"],
            "plan": state["plan"],
            "current_task": None,  # Clear current task
            "tool_result": state.get("tool_result"),
            "conversation_history": state["conversation_history"],
            "plan_id": state["plan_id"],
            "use_llm_tool_selection": True  # Ensure LLM tool selection continues
        }
    
    def _check_completion(self, state: AgentState) -> Dict:
        """Process state before checking completion."""
        # This is a processing node that simply passes the state through
        # The actual routing decision is made by _should_continue
        return {
            "user_query": state["user_query"],
            "plan": state["plan"],
            "current_task": state.get("current_task"),
            "tool_result": state.get("tool_result"),
            "conversation_history": state["conversation_history"],
            "plan_id": state["plan_id"]
        }
    
    def _route_after_task_execution(self, state: AgentState) -> str:
        """Route after task execution."""
        if state.get("pending_tool_execution"):
            return "awaiting_confirmation"
        return "continue"
    
    def _route_after_human_confirmation(self, state: AgentState) -> str:
        """Route after human confirmation based on whether to execute the tool or cancel."""
        # In a real implementation, this would check the user's confirmation decision
        # For now, we'll simulate user approval
        print("Routing after human confirmation...")
        return "execute_tool"
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if the agent should continue or complete the troubleshooting process."""
        # Check if all tasks are completed
        #tasks = state["plan"].get("tasks", [])
        #all_completed = all(task.get("status") == "completed" for task in tasks)
        continue_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant helping to troubleshoot a user query.
        The current plan which was modified is: {plan}
        The user query is: {user_query}
        The previous tool result is: {tool_result}
        The observation is: {observation}
        Should the agent continue troubleshooting? (ONLY REPLY yes/no)!
        """)
        chain = continue_prompt | self.llm | StrOutputParser()
        continue_decision = chain.invoke({
            "plan": state["plan"],
            "user_query": state["user_query"],
            "tool_result": state.get("tool_result"),
            "observation": state.get("observation")
        })
        self.console.print("Continue Decision:", continue_decision)

        return "complete" if continue_decision.lower() == "no" else "continue"
    
    def _report_result(self, state: AgentState) -> Dict:
        """Report the final troubleshooting result."""
        # Generate summary
        summary = f"Troubleshooting completed for: {state['user_query']}\n\n"
        summary += "Tasks completed:\n"
        for task in state["plan"].get("tasks", []):
            summary += f"- {task['description']}: {task.get('status', 'pending')}\n"
        
        # Use report_result tool
        report_tool = tools.get("report_result")
        result = report_tool(summary=summary, result="Troubleshooting process completed.")
        
        return {
            "user_query": state["user_query"],
            "plan": state["plan"],
            "current_task": state.get("current_task"),
            "tool_result": result,
            "conversation_history": state["conversation_history"],
            "plan_id": state["plan_id"]
        }
    
    def generate_plan(self, user_query: str) -> Dict[str, Any]:
        """Generate a troubleshooting plan without executing it."""
        import uuid
        plan_id = str(uuid.uuid4())
        
        initial_state = {
            "user_query": user_query,
            "plan": {},
            "current_task": None,
            "tool_result": None,
            "conversation_history": [],
            "plan_id": plan_id
        }
        
        # Create a simplified graph that only generates the plan
        plan_state = self._create_plan(initial_state)
        return {
            "plan_id": plan_state["plan_id"],
            "plan": plan_state["plan"]
        }
    
    def execute_plan(self, user_query: str, plan: Dict) -> Dict[str, Any]:
        """Execute a pre-generated plan."""
        import uuid
        plan_id = str(uuid.uuid4())
        
        # Mark all tasks as pending initially
        for task in plan.get("tasks", []):
            if "status" not in task:
                task["status"] = "pending"
        
        initial_state = {
            "user_query": user_query,
            "plan": plan,
            "current_task": None,
            "tool_result": None,
            "conversation_history": [],
            "plan_id": plan_id
        }
        result = self.graph.invoke(initial_state, config={"recursion_limit": 200})
        return {
            "plan_id": result["plan_id"],
            "plan": result["plan"],
            "final_result": result["tool_result"]
        }

    def print_mermaid_workflow(self):
        """
        Utility: print Mermaid diagram to visualize the graph edges.
        """
        try:
            mermaid = self.graph.get_graph().draw_mermaid_png(
                output_file_path="agent_workflow.png",
                max_retries=5,
                retry_delay=2,
            )
        except Exception as e:
            print(f"Error generating mermaid PNG: {e}")
            mermaid = self.graph.get_graph().draw_mermaid()
            self.console.print(
                Panel.fit(
                    Syntax(mermaid, "mermaid", theme="monokai", line_numbers=False),
                    title="Agent Workflow (Mermaid)",
                    border_style="cyan",
                )
            )
            print(self.graph.get_graph().draw_ascii())