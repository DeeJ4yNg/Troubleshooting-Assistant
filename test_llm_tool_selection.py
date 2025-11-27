#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试LLM工具选择功能
"""

import os
import sys
from langchain_openai import ChatOpenAI
from agent.react_agent import ReactAgent
from tools import available_tools
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_llm_tool_selection():
    """测试LLM工具选择功能"""
    print("开始测试LLM工具选择功能...")
    
    # 初始化LLM
    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=0.7
    )
    
    # 初始化工具
    tools_dict = {name: tool for name, tool in available_tools}
    
    # 初始化代理
    agent = ReactAgent(llm=llm, tools=tools_dict)
    
    # 测试查询
    test_query = "Excel右键菜单不显示，如何解决？"
    print(f"\n测试查询: {test_query}")
    
    try:
        # 创建初始状态
        initial_state = {
            "user_query": test_query,
            "plan": {"tasks": []},
            "conversation_history": [
                {"role": "user", "content": test_query}
            ],
            "plan_id": f"plan_test_{os.urandom(8).hex()}",
            "use_llm_tool_selection": True  # 启用LLM工具选择
        }
        
        # 运行代理
        print("\n启动代理，使用LLM进行工具选择...")
        result = agent.graph.invoke(initial_state)
        
        # 输出结果
        print("\n代理执行完成，结果:")
        print(f"- 完成的任务数: {len(result.get('plan', {}).get('tasks', []))}")
        print(f"- 最终工具结果: {result.get('tool_result')}")
        
        # 检查是否正确使用了LLM工具选择
        plan_tasks = result.get('plan', {}).get('tasks', [])
        llm_selected_tasks = [task for task in plan_tasks if 'task_llm_' in task.get('task_id', '')]
        
        print(f"\n- LLM选择的任务数: {len(llm_selected_tasks)}")
        for task in llm_selected_tasks:
            print(f"  * {task.get('task_id')}: {task.get('description')} -> 使用工具: {task.get('tool')}")
        
        print("\nLLM工具选择功能测试完成!")
        return True
        
    except Exception as e:
        print(f"\n测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_tool_selection()
    sys.exit(0 if success else 1)
