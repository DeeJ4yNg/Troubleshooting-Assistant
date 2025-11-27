#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the fixes to the Windows Troubleshooting Agent.
"""

import sys
import os

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

def main():
    """Run all tests."""
    print("Testing fixes to Windows Troubleshooting Agent...\n")
    
    tests = [
        test_agent_initialization,
        test_state_definition
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! The fixes are working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())