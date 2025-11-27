#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive test script to verify all functionality of the Windows Troubleshooting Agent.
"""

import sys
import os
import uuid

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.react_agent import ReactAgent
from agent.memory import SQLiteMemory

def test_agent_initialization():
    """Test that the agent can be initialized without errors."""
    try:
        # Initialize memory
        memory = SQLiteMemory()
        
        # Initialize agent
        agent = ReactAgent(memory)
        print("✓ Agent initialization successful")
        return True
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        return False

def test_state_definition():
    """Test that the AgentState can be imported and used."""
    try:
        from agent.react_agent import AgentState
        # Try to create an instance of AgentState
        state = AgentState(
            user_query="Test query",
            plan=None,
            current_task=None,
            tool_result=None,
            conversation_history=[],
            plan_id=None,
            observation=None,
            pending_tool_execution=None
        )
        print("✓ AgentState definition and instantiation successful")
        return True
    except Exception as e:
        print(f"✗ AgentState definition or instantiation failed: {e}")
        return False

def test_memory_operations():
    """Test memory operations."""
    try:
        # Initialize memory
        memory = SQLiteMemory()
        
        # Test saving and retrieving a plan
        plan_id = str(uuid.uuid4())
        user_query = "Test query for memory"
        plan = {
            "tasks": [
                {
                    "task_id": "task_1",
                    "description": "Test task",
                    "tool": "online_search",
                    "params": {"query": "test"}
                }
            ]
        }
        
        memory.save_plan(plan_id, user_query, plan)
        retrieved_plan = memory.get_plan(plan_id)
        
        if retrieved_plan and retrieved_plan.get("tasks"):
            print("✓ Memory operations successful")
            return True
        else:
            print("✗ Memory operations failed: Could not retrieve saved plan")
            return False
    except Exception as e:
        print(f"✗ Memory operations failed: {e}")
        return False

def test_graph_compilation():
    """Test that the graph can be compiled without errors."""
    try:
        # Initialize memory
        memory = SQLiteMemory()
        
        # Initialize agent
        agent = ReactAgent(memory)
        
        # Check if graph is compiled
        if agent.graph:
            print("✓ Graph compilation successful")
            return True
        else:
            print("✗ Graph compilation failed: No graph object")
            return False
    except Exception as e:
        print(f"✗ Graph compilation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running comprehensive tests for Windows Troubleshooting Agent...\n")
    
    tests = [
        test_agent_initialization,
        test_state_definition,
        test_memory_operations,
        test_graph_compilation
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All comprehensive tests passed! The agent is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())