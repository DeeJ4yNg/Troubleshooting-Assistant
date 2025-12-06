from typing import Dict, Any, List, Optional, Sequence, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.tools import StructuredTool, render_text_description
from pydantic.v1 import BaseModel, Field
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
    def __init__(self, memory: SQLiteMemory, ui=None):
        self.memory = memory
        # Initialize OpenAI LLM with API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        api_base = os.getenv("OPENAI_API_BASE")
        self.console = Console()
        self.ui = ui
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.llm = ChatOpenAI(
            base_url=api_base,
            model=openai_model,
            openai_api_key=openai_api_key,
            temperature=0.7
        )
        self.graph = self._build_graph()
        
        # Ask user if they want to clean memory
        if self.ui:
            self._ask_to_clean_memory()
    
    def _ask_to_clean_memory(self):
        """Ask user if they want to clean the memory."""
        from rich.prompt import Confirm
        try:
            if Confirm.ask("[bold red]Do you want to clean the agent memory before starting?[/bold red]", default=False):
                self.memory.clear_memory()
                self.console.print("[green]Memory cleaned successfully.[/green]")
        except Exception as e:
            self.console.print(f"[red]Error cleaning memory: {e}[/red]")

    def _get_langchain_tools(self) -> List[StructuredTool]:
        """Convert agent tools to LangChain StructuredTool objects."""
        langchain_tools = []
        for name, tool in tools.items():
            if hasattr(tool, "name") and hasattr(tool, "description") and hasattr(tool, "__call__"):
                # Create StructuredTool from the tool's __call__ method
                # We need to access the __call__ method of the instance
                func = tool.__call__
                
                # Create the tool
                l_tool = StructuredTool.from_function(
                    func=func,
                    name=name,
                    description=tool.description
                )
                langchain_tools.append(l_tool)
        return langchain_tools

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
        
        # Define Pydantic models for structured output
        class Task(BaseModel):
            task_id: str = Field(description="Unique identifier for the task, e.g., 'task_1'")
            description: str = Field(description="Clear description of what the task does")
            tool: str = Field(description="The appropriate tool to use")
            params: Dict[str, Any] = Field(description="Proper parameters for the tool")
            status: str = Field(description="Initial status of the task", default="pending")
            result: Optional[str] = Field(description="Result of the task execution (success/failure)", default=None)
            reason: Optional[str] = Field(description="Reason for the result or status", default=None)

        class Plan(BaseModel):
            tasks: List[Task] = Field(description="List of troubleshooting tasks")

        # Set up the parser
        parser = JsonOutputParser(pydantic_object=Plan)

        # Get tools and generate description using LangChain
        langchain_tools = self._get_langchain_tools()
        tool_descriptions = render_text_description(langchain_tools)

        prompt = ChatPromptTemplate.from_template("""
        You are a Windows OS troubleshooting expert. Create a detailed step-by-step plan to troubleshoot the following issue:
        
        {user_query}
        
        The plan should include multiple specific tasks that can be executed using the following tools:
        {tool_descriptions}
        
        Based on the knowledge you have and the tools available, create a comprehensive plan with ONLY 3-5 tasks that thoroughly address the issue.
        
        {format_instructions}
        
        Remember:
        1. Each task must have a unique task_id.
        2. The tool parameter must be one of the available tools.
        3. The params parameter must contain the correct parameters for the selected tool.
        4. Each task must have a clear description.
        5. The plan must be executable using the available tools.
        6. Always try to search for relevant information before executing any scripts.
        """)
        
        if state.get("plan"):
            return state

        chain = prompt | self.llm | parser
        
        try:
            plan = chain.invoke({
                "user_query": state["user_query"],
                "format_instructions": parser.get_format_instructions(),
                "tool_descriptions": tool_descriptions
            })
            print(f"Generated plan: {plan}")  # Debug print
            
            # Ensure all tasks have a status (double check)
            if "tasks" in plan:
                for task in plan["tasks"]:
                    if "status" not in task:
                        task["status"] = "pending"
                    if "result" not in task:
                        task["result"] = "N/A"
                    if "reason" not in task:
                        task["reason"] = "N/A"
            
        except Exception as e:
            print(f"Error generating plan: {e}")
            # Fallback plan
            plan = {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "description": "Search for information about the issue",
                        "tool": "online_search",
                        "status": "pending",
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
        current_task = pending_execution.get("task")
        
        # Display tool execution if UI is available
        if self.ui and current_task:
            self.ui.display_tool_execution(current_task)
        
        confirm = Prompt.ask(f"[bold yellow]Are you sure you want to execute tool: {tool_name} with params: {params}? (yes/no)[/bold yellow]")
        if confirm.lower() != "yes":
            console.print("[bold red]Tool execution cancelled by user[/bold red]")
            exit(0)
            #return state
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
        
        # Mark task as completed immediately after execution
        if pending_execution.get("task"):
            task_id = pending_execution["task"]["task_id"]
            for task in state["plan"].get("tasks", []):
                if task["task_id"] == task_id:
                    task["status"] = "completed"
                    #break
                    
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
            You are a Windows OS troubleshooting expert. Based on the troubleshooting plan and the conversation history, analyze the following tool execution result and suggest 
            any necessary modifications to the troubleshooting plan.
            
            IMPORTANT: Avoid creating duplicate tasks that were already executed in the conversation history. 
            DO NOT fall into infinite loops by repeatedly executing similar tools or asking similar questions.
            Focus on making progress toward resolving the issue rather than repeating previous steps.
            
            Original user query: {user_query}

            Troubleshooting plan:
            {plan}
            
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
            - run_cmd_command: Execute a Windows CMD command (Example params: {{"command": "ipconfig /all"}})
            - online_search: Search the internet (Example params: {{"query": "Windows 11 Settings app not opening fix", "max_results": 3}})
            - knowledge_retrieval: Retrieve knowledge from database (Example params: {{"query": "Windows Settings app repair methods", "limit": 5}})
            - check_registry_key: Return all values and subkeys in a registry key (Example params: {{"key_path": "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion"}})
            - modify_registry_key: Add, remove, or modify registry keys and values (Example params: {{"key_path": "HKCU\\Software\\Test", "operation": "set_value", "value_name": "TestVal", "value_data": "1", "value_type": "REG_SZ"}})
            
            Keep modifications minimal and focused. If no changes are needed, set should_modify_plan to false.
            Always prioritize making forward progress and avoiding repetition.

            REMEMBER: 
            1.NO DUPLICATE TASKS IN THE PLAN!
            2.All tasks in the plan should be executable by the available tools.
            3.If a task is not executable, remove it from the plan!
            4.DO NOT REMOVE OR MODIFY THE CURRENT TASK FROM THE PLAN!!!
            5.DO NOT REMOVE OR MODIFY THE TASKS THAT ARE COMPLETED!!!
            6.DO NOT MODIFY THE RESULT AND REASON FOR THE TASKS THAT ARE NOT COMPLETED YET!!! 
            7.All plan modifications must be added to "modifications" list!!!
            8.Respond with a JSON object containing:
            {{
                "success": true/false,
                "analysis": "Brief analysis of the result, do not contain any plan modifications.",
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
            """)
            
            plan_tasks_info = "\n".join([
                f"- {task.get('task_id', 'N/A')}: {task.get('description', 'N/A')} (Status: {task.get('status', 'pending')})"
                for task in state.get("plan", {}).get("tasks", [])
            ])
            
            chain = analysis_prompt | self.llm | StrOutputParser()
            
            # Format conversation history for the prompt
            conversation_history_text = "\n".join([
                f"Task {msg.get('task_id', 'N/A')} ({msg.get('tool', 'N/A')}): {msg['content']}" 
                for msg in state.get("conversation_history", [])[-10:]  # Include last 10 messages to keep context manageable
            ])
            
            with self.console.status("[bold green]Analyzing tool result...[/bold green]") as status:
                analysis_result = chain.invoke({
                "user_query": state["user_query"],
                "conversation_history": conversation_history_text,
                "task_description": task_description,
                "tool_result": str(tool_result),
                "plan_tasks": plan_tasks_info,
                "plan": state.get("plan", {})
                })
            
            # Try to parse the LLM response as JSON
            import json
            import re
            try:
                # Step 1: Extract JSON from markdown code blocks if present
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', analysis_result)
                if json_match:
                    plan_str = json_match.group(1)
                else:
                    # Step 2: Look for JSON object directly if no code blocks
                    json_match = re.search(r'\{[\s\S]*\}', analysis_result)
                    if json_match:
                        plan_str = json_match.group(0)
                    else:
                        plan_str = analysis_result
                
                # Step 3: Clean up the JSON string
                plan_str = plan_str.strip()  # Remove leading/trailing whitespace
                plan_str = re.sub(r'^\s*#.*$', '', plan_str, flags=re.MULTILINE)  # Remove comments
                plan_str = re.sub(r',\s*([}\]])', r'\1', plan_str)  # Remove trailing commas
                
                # Step 4: Try to parse the cleaned JSON
                analysis_json = json.loads(plan_str)
                self.console.print(analysis_json)
                observation["success"] = analysis_json.get("success", False)
                observation["llm_analysis"] = analysis_json.get("analysis", "")
                observation["should_modify_plan"] = analysis_json.get("should_modify_plan", False)
                observation["plan_modifications"] = analysis_json.get("modifications", [])
            except json.JSONDecodeError as e:
                self.console.print(f"JSON parsing failed for analysis result: {e}")
                self.console.print(f"Attempted to parse: {plan_str}")
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
            "task_id": current_task.get("task_id", "unknown"),
            "description": current_task.get("description", ""),
            "tool": current_task.get("tool", ""),
            "params": current_task.get("params", {}),
            "content": f"Tool_Result: {tool_result}. Plan Modifications: {observation.get('plan_modifications', '')}. LLM Analysis: {observation.get('llm_analysis', '')}"
        })
        
        # Display observation if UI is available
        if self.ui:
            #self.ui.display_observation(observation)
            self.ui.display_plan_modifications(observation)

        self._print_agent_context(state)
        
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
        # Note: Task status is already marked as completed in _execute_confirmed_tool
        
        # Update the executed task with detailed result and reason from observation
        current_task_in_state = state.get("current_task")
        observation = state.get("observation", {})
        
        if current_task_in_state and observation:
             task_id = current_task_in_state.get("task_id")
             plan_tasks = state["plan"].get("tasks", [])
             for task in plan_tasks:
                 if task.get("task_id") == task_id:
                     # Update status just in case, and add result/reason
                     task["status"] = "completed"
                     task["result"] = "success" if observation.get("success", False) else "failure"
                     
                     # Use LLM analysis or details as reason
                     llm_analysis = observation.get("llm_analysis", "")
                     reason_text = ""
                     if isinstance(llm_analysis, dict):
                        reason_text = llm_analysis.get("analysis", str(llm_analysis))
                     else:
                        reason_text = str(llm_analysis) if llm_analysis else str(observation.get("details", ""))
                     
                     task["reason"] = f"{reason_text[:500]}..." # Truncate if too long"
                     break
        
        # Fall back to existing observation-based modifications if no tool result or LLM fails
        if state.get("observation", {}).get("should_modify_plan", False) or state.get("observation", {}).get("should_modify_plan", {}):
            modifications = state.get("observation", {}).get("plan_modifications", [])
            plan_tasks = state["plan"].get("tasks", [])
            
            for modification in modifications:
                mod_type = modification.get("type")
                task_id = modification.get("task_id")
                reason = modification.get("reason")
                
                if mod_type == "add" and "new_task" in modification:
                    # Add new task
                    new_task = modification["new_task"]
                    # Detect if the new task is a duplicated one, if yes, skip it.
                    if any(task.get("task_id") == new_task.get("task_id") for task in plan_tasks):
                        self.console.print(f"[bold yellow]Duplicated task {new_task.get('task_id', 'N/A')}. Task: {new_task.get('description', 'N/A')} - Reason: {reason}[/bold yellow]")
                        continue

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
                    if new_task["tool"] == "knowledge_retrieval":
                        # Make sure params is a dictionary
                        if not isinstance(new_task["params"], dict):
                            new_task["params"] = {}
                        if "query" not in new_task["params"]:
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
                            if task.get("tool") == "knowledge_retrieval":
                                if "params" not in task:
                                    task["params"] = {}
                                # Make sure params is a dictionary
                                elif not isinstance(task["params"], dict):
                                    task["params"] = {}
                                if "query" not in task["params"]:
                                    task["params"]["query"] = state.get("user_query", "")

                            # 如果有 new_task，则修改当前 task_id 对应的 description、tool、params，并将 status 设为 pending
                            if "new_task" in modification:
                                new_task_data = modification["new_task"]
                                
                                if "description" in new_task_data:
                                    task["description"] = new_task_data["description"]
                                if "tool" in new_task_data:
                                    task["tool"] = new_task_data["tool"]
                                if "params" in new_task_data:
                                    task["params"] = new_task_data["params"]
                                task["status"] = "pending"  # 重置状态为 pending
                            
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
        table.add_column("Result")
        table.add_column("Reason")
        for task in state["plan"]["tasks"]:
            table.add_row(
                task["task_id"],
                task.get("description", "N/A"),
                task.get("tool", "N/A"),
                str(task.get("params", {})),
                task.get("status", "N/A"),
                task.get("result", "N/A"),
                task.get("reason", "N/A")
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
        print("Checking completion status with current plan...")
        print(state["plan"])
        continue_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant helping to troubleshoot a user query.
        The current plan which was modified is: {plan}
        The user query is: {user_query}
        The previous tool result is: {tool_result}
        The observation is: {observation}
        DO NOT reply no if you see any task status is pending!
        If all task status is completed, check if the tool result is valid without any error, if not, REPLY yes to continue troubleshooting!
        If you think you have done all the tasks and all things you can do but not sure if the issue is resolved or not, REPLY no to complete the troubleshooting process and user can check the result!
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
    
    def _print_agent_context(self, state: AgentState, step: str = "") -> None:
        """Print the current agent context for debugging and testing purposes."""
        self.console.print("\n[bold green]=== Agent Context Information ===[/bold green]")
        self.console.print(f"[bold]Step:[/bold] {step}")
        self.console.print(f"[bold]Plan ID:[/bold] {state.get('plan_id')}")
        self.console.print(f"[bold]User Query:[/bold] {state.get('user_query')}")
        
        if state.get('current_task'):
            task = state['current_task']
            self.console.print(f"[bold]Current Task:[/bold] {task.get('task_id')} - {task.get('description')}")
            self.console.print(f"[bold]Task Tool:[/bold] {task.get('tool')}")
            self.console.print(f"[bold]Task Params:[/bold] {task.get('params')}")
        
        if state.get('tool_result'):
            self.console.print(f"[bold]Last Tool Result:[/bold] {state.get('tool_result')}")
        
        if state.get('observation'):
            obs = state.get('observation')
            if isinstance(obs, dict):
                self.console.print("[bold]Observation:[/bold]")
                for k, v in obs.items():
                    self.console.print(f"  [cyan]{k}:[/cyan] {v}")
            else:
                self.console.print(f"[bold]Observation:[/bold] {obs}")
        
        if state.get('pending_tool_execution'):
            pending_exec = state['pending_tool_execution']
            self.console.print(f"[bold]Pending Tool Execution:[/bold] {pending_exec.get('tool_name')} with params {pending_exec.get('params')}")
        
        # Print plan status
        if state.get('plan') and state['plan'].get('tasks'):
            self.console.print("\n[bold]Plan Status:[/bold]")
            for task in state['plan']['tasks']:
                status = task.get('status', 'pending')
                self.console.print(f"  - {task.get('task_id')}: {status} - {task.get('description')[:50]}...")

        if state.get('conversation_history'):
            # 格式化显示对话历史
            conversation_history = state.get("conversation_history", [])
            if conversation_history:
                self.console.print("\n[bold]Conversation history:[/bold]")
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Task ID", style="dim", width=12)
                table.add_column("Description")
                table.add_column("Tool")
                table.add_column("Params")
                table.add_column("Content")
                for msg in conversation_history:
                    task_id = msg.get("task_id", "unknown")
                    description = msg.get("description", "")
                    tool = msg.get("tool", "")
                    params = str(msg.get("params", ""))
                    content = msg.get("content", "")
                    table.add_row(task_id, description, tool, params, content)
                self.console.print(table)
            else:
                self.console.print("\n[bold]Conversation history:[/bold] (empty)")
        
        self.console.print("[bold green]=================================[/bold green]\n")
    
    def execute_plan(self, user_query: str, plan: Dict) -> Dict[str, Any]:
        """Execute a pre-generated plan with context tracking."""
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
        
        # Print initial context
        self._print_agent_context(initial_state, "Initial State")
        
        result = self.graph.invoke(initial_state, config={"recursion_limit": 200})
        
        # Print final context
        self._print_agent_context(result, "Final State")
        
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