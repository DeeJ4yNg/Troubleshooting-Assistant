import os
import re
import subprocess
from typing import List, Dict, Any
try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS
from agent.memory import SQLiteMemory

class ListLogFilesTool:
    name = "list_log_files"
    description = "List all log files in a specified directory path"
    
    def __call__(self, path: str) -> List[str]:
        """List all log files in the specified directory."""
        try:
            if not os.path.exists(path):
                return [f"Error: Directory '{path}' does not exist"]
            
            log_files = []
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(('.log', '.txt', '.evt', '.evtx')):
                        log_files.append(os.path.join(root, file))
            
            if not log_files:
                return [f"No log files found in '{path}'"]
            
            return log_files
        except Exception as e:
            return [f"Error: {str(e)}"]

class ReadErrorLogsTool:
    name = "read_error_logs"
    description = "Read and extract error messages from a log file"
    
    def __call__(self, file_path: str, max_lines: int = 100) -> List[str]:
        """Read error messages from the specified log file."""
        try:
            if not os.path.exists(file_path):
                return [f"Error: File '{file_path}' does not exist"]
            
            errors = []
            error_patterns = [
                r'ERROR', r'Error', r'error',
                r'FAILED', r'Failed', r'failed',
                r'EXCEPTION', r'Exception', r'exception',
                r'CRITICAL', r'Critical', r'critical',
                r'WARNING', r'Warning', r'warning'
            ]
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for line in lines[-max_lines:]:
                if any(re.search(pattern, line) for pattern in error_patterns):
                    errors.append(line.strip())
            
            if not errors:
                return [f"No error messages found in the last {max_lines} lines of '{file_path}'"]
            
            return errors
        except Exception as e:
            return [f"Error: {str(e)}"]

class WritePS1FileTool:
    name = "write_ps1_file"
    description = "Create a PowerShell script file with the given content"
    
    def __call__(self, file_path: str, content: str) -> str:
        """Create a PowerShell script file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully created PowerShell script at '{file_path}'"
        except Exception as e:
            return f"Error: {str(e)}"

class RunPS1TestTool:
    name = "run_ps1_test"
    description = "Run a PowerShell script and return the output"
    
    def __call__(self, file_path: str) -> str:
        """Run a PowerShell script and return the output."""
        try:
            if not os.path.exists(file_path):
                return f"Error: File '{file_path}' does not exist"
            
            result = subprocess.run(
                ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = f"Exit Code: {result.returncode}\n"
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Script execution timed out after 30 seconds"
        except Exception as e:
            return f"Error: {str(e)}"

class OnlineSearchTool:
    name = "online_search"
    description = "Search the internet using DuckDuckGo for the given query"
    
    def __call__(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search the internet using DuckDuckGo."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "content": result.get("body", "")
                })
            
            return formatted_results
        except Exception as e:
            return [{"error": str(e)}]

class KnowledgeRetrievalTool:
    name = "knowledge_retrieval"
    description = "Retrieve knowledge from the SQLite database using RAG"
    
    def __init__(self, memory: SQLiteMemory):
        self.memory = memory
    
    def __call__(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Retrieve knowledge from the SQLite database."""
        try:
            results = self.memory.retrieve_knowledge(query, limit=limit)
            return results
        except Exception as e:
            return [{"error": str(e)}]

class ReportResultTool:
    name = "report_result"
    description = "Report the final troubleshooting result and summary to the user"
    
    def __call__(self, summary: str, result: str) -> str:
        """Report the troubleshooting result."""
        try:
            return f"# Troubleshooting Result\n\n## Summary\n{summary}\n\n## Final Result\n{result}"
        except Exception as e:
            return f"Error: {str(e)}"

# Create tool instances
memory = SQLiteMemory()

list_log_files_tool = ListLogFilesTool()
read_error_logs_tool = ReadErrorLogsTool()
write_ps1_file_tool = WritePS1FileTool()
run_ps1_test_tool = RunPS1TestTool()
online_search_tool = OnlineSearchTool()
knowledge_retrieval_tool = KnowledgeRetrievalTool(memory)
report_result_tool = ReportResultTool()

# Tool registry for easy access
tools = {
    "list_log_files": list_log_files_tool,
    "read_error_logs": read_error_logs_tool,
    "write_ps1_file": write_ps1_file_tool,
    "run_ps1_test": run_ps1_test_tool,
    "online_search": online_search_tool,
    "knowledge_retrieval": knowledge_retrieval_tool,
    "report_result": report_result_tool
}
