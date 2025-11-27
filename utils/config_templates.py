#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration Templates for LangGraph Agent Framework

This module provides template configurations for generating different types of
LangGraph agent frameworks with predefined workflows.
"""

from typing import Dict, List, Any

# Template configurations
TEMPLATES = {
    "basic_react": {
        "name": "Basic ReAct Agent",
        "description": "A simple ReAct-style agent with planning, execution, and observation phases.",
        "nodes": [
            {"name": "create_plan", "method_name": "_create_plan"},
            {"name": "execute_task", "method_name": "_execute_task"},
            {"name": "observe_result", "method_name": "_observe_result"}
        ],
        "edges": [
            {"source": "START", "target": "create_plan"},
            {"source": "create_plan", "target": "execute_task"},
            {"source": "execute_task", "target": "observe_result"},
            {"source": "observe_result", "target": "execute_task", "condition": "_should_continue"}
        ]
    },
    "human_in_the_loop": {
        "name": "Human-in-the-Loop Agent",
        "description": "An agent that requires human confirmation before executing tools.",
        "nodes": [
            {"name": "create_plan", "method_name": "_create_plan"},
            {"name": "execute_task", "method_name": "_execute_task"},
            {"name": "get_human_confirmation", "method_name": "_get_human_confirmation"},
            {"name": "execute_confirmed_tool", "method_name": "_execute_confirmed_tool"},
            {"name": "observe_result", "method_name": "_observe_result"}
        ],
        "edges": [
            {"source": "START", "target": "create_plan"},
            {"source": "create_plan", "target": "execute_task"},
            {"source": "execute_task", "target": "get_human_confirmation"},
            {"source": "get_human_confirmation", "target": "execute_confirmed_tool"},
            {"source": "execute_confirmed_tool", "target": "observe_result"},
            {"source": "observe_result", "target": "execute_task", "condition": "_should_continue"}
        ]
    },
    "llm_tool_selection": {
        "name": "LLM Tool Selection Agent",
        "description": "An agent that uses LLM to intelligently select tools based on context.",
        "nodes": [
            {"name": "create_plan", "method_name": "_create_plan"},
            {"name": "select_tool_with_llm", "method_name": "_select_tool_with_llm"},
            {"name": "execute_task", "method_name": "_execute_task"},
            {"name": "execute_confirmed_tool", "method_name": "_execute_confirmed_tool"},
            {"name": "observe_result", "method_name": "_observe_result"},
            {"name": "handle_user_response", "method_name": "_handle_user_response"}
        ],
        "edges": [
            {"source": "START", "target": "create_plan"},
            {"source": "create_plan", "target": "select_tool_with_llm"},
            {"source": "select_tool_with_llm", "target": "execute_task"},
            {"source": "execute_task", "target": "execute_confirmed_tool", "condition": "_should_get_confirmation"},
            {"source": "execute_confirmed_tool", "target": "observe_result"},
            {"source": "observe_result", "target": "select_tool_with_llm", "condition": "_should_continue"},
            {"source": "execute_task", "target": "handle_user_response", "condition": "_has_pending_question"},
            {"source": "handle_user_response", "target": "select_tool_with_llm"}
        ]
    }
}

# Default implementation templates for common methods
IMPLEMENTATION_TEMPLATES = {
    "create_plan": """
    def _create_plan(self, state: AgentState) -> AgentState:
        """Create an initial plan based on user query."""
        user_query = state.get("user_query", "")
        
        # Generate plan using LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that creates structured plans for solving problems."),
            ("user", "Create a step-by-step plan to solve the following problem:\n{query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        plan_text = chain.invoke({"query": user_query})
        
        # Parse the plan into tasks
        tasks = []
        for i, line in enumerate(plan_text.split('\n')):
            if line.strip() and (line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '- ', '* '))):
                task_id = f"task_{i+1}"
                description = line.strip().lstrip('1234567890. -*').strip()
                tasks.append({
                    "task_id": task_id,
                    "description": description,
                    "status": "pending"
                })
        
        state["plan"] = {"tasks": tasks}
        state["plan_id"] = f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return state
""",
    
    "execute_task": """
    def _execute_task(self, state: AgentState) -> AgentState:
        """Execute the next task in the plan."""
        plan = state.get("plan", {})
        tasks = plan.get("tasks", [])
        
        # Find the next pending task
        pending_task = None
        for task in tasks:
            if task.get("status") == "pending":
                pending_task = task
                break
        
        if not pending_task:
            return state
        
        # Set current task
        state["current_task"] = pending_task
        pending_task["status"] = "in_progress"
        
        # Check if tool selection should use LLM
        if state.get("use_llm_tool_selection", False):
            # Tool selection will be handled by the LLM tool selection node
            return state
        
        # Default tool selection logic
        # This should be customized based on your specific tools and tasks
        
        return state
""",
    
    "observe_result": """
    def _observe_result(self, state: AgentState) -> AgentState:
        """Observe and analyze tool results."""
        tool_result = state.get("tool_result")
        current_task = state.get("current_task")
        plan = state.get("plan", {})
        tasks = plan.get("tasks", [])
        
        if current_task:
            # Update current task status
            current_task["status"] = "completed"
            current_task["result"] = tool_result
        
        # Analyze results and potentially add new tasks
        if tool_result:
            # This could use LLM to analyze results and generate new tasks
            # For now, we'll just mark the workflow to continue
            state["use_llm_tool_selection"] = True  # Ensure LLM selection continues
        
        # Update conversation history
        conversation_history = state.get("conversation_history", [])
        if tool_result:
            conversation_history.append({
                "role": "assistant",
                "content": f"Tool result: {str(tool_result)}"
            })
        
        return state
""",
    
    "should_continue": """
    def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue the workflow or finish."""
        plan = state.get("plan", {})
        tasks = plan.get("tasks", [])
        
        # Check if there are any pending tasks
        pending_tasks = [task for task in tasks if task.get("status") == "pending"]
        
        if pending_tasks:
            return "continue"
        else:
            return "finish"
"""
}

def get_template(template_name: str) -> Dict[str, Any]:
    """
    Get a predefined template configuration.
    
    Args:
        template_name: Name of the template to retrieve
        
    Returns:
        Template configuration dictionary
        
    Raises:
        ValueError: If the template name is not found
    """
    if template_name not in TEMPLATES:
        raise ValueError(f"Template '{template_name}' not found. Available templates: {list(TEMPLATES.keys())}")
    
    return TEMPLATES[template_name]

def get_implementation_template(method_name: str) -> str:
    """
    Get a predefined implementation template for a method.
    
    Args:
        method_name: Name of the method to retrieve
        
    Returns:
        Method implementation template
    """
    return IMPLEMENTATION_TEMPLATES.get(method_name, """
    def {method_name}(self, state: AgentState) -> AgentState:
        """Default implementation."""
        return state
""".format(method_name=method_name))

def list_templates() -> List[str]:
    """
    List all available templates.
    
    Returns:
        List of template names
    """
    return list(TEMPLATES.keys())

def describe_template(template_name: str) -> str:
    """
    Get a description of a template.
    
    Args:
        template_name: Name of the template
        
    Returns:
        Template description
        
    Raises:
        ValueError: If the template name is not found
    """
    template = get_template(template_name)
    return f"{template['name']}: {template['description']}"
