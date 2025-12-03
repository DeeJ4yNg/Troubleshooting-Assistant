import os
import re
import subprocess
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from ddgs import DDGS
from agent.memory import SQLiteMemory
from agent.knowledge_manager import KnowledgeManager

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ListLogFilesTool:
    name = "list_log_files"
    description = "List all log files in a specified directory path. Example: {{\"path\": \"C:/Windows/Logs\"}} to list all log files in Windows logs directory."
    
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
    description = "Read and extract error messages from a log file. Example: {{\"file_path\": \"C:/Windows/Logs/System.evtx\", \"max_lines\": 100}} to read errors from system event log."
    
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
    description = "Create a PowerShell script file with the given content. Example: {{\"file_path\": \"C:/temp/fix_settings.ps1\", \"content\": \"Get-AppxPackage Microsoft.Windows.SettingsApp | Reset-AppxPackage\"}} to create a script that resets Settings app."
    
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
    description = "Run a PowerShell script and return the output. Example: {{\"file_path\": \"C:/temp/fix_settings.ps1\"}} to execute the previously created PowerShell script."
    
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
    description = "Search the internet using DuckDuckGo for the given query. Example: {{\"query\": \"Windows 11 Settings app not opening fix\", \"max_results\": 3}} to search for troubleshooting solutions."
    
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
    description = "Retrieve knowledge from the knowledge base using hybrid search (BM25 + ANN). Example: {{'query': 'Windows Settings app repair methods', 'limit': 5}} to retrieve relevant information."
    
    def __init__(self):
        """初始化知识库检索工具。"""
        try:
            # 使用KnowledgeManager代替SQLiteMemory
            self.knowledge_manager = KnowledgeManager()
            # 尝试导入嵌入模型
            self.embedding_model = self._init_embedding_model()
            logger.info("KnowledgeRetrievalTool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeRetrievalTool: {e}")
            # 创建一个简单的备份实现，只支持关键词搜索
            self.knowledge_manager = None
    
    def _init_embedding_model(self) -> Optional[Any]:
        """初始化嵌入模型"""
        try:
            # 尝试导入OpenAI模型（如果可用）
            try:
                import openai
                # 检查API密钥
                if os.environ.get("OPENAI_API_KEY"):
                    return "openai"
            except ImportError:
                pass
            
            # 尝试导入其他嵌入模型库
            # 这里可以添加其他嵌入模型的支持
            
            return None
        except Exception:
            return None
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成文本的嵌入向量"""
        try:
            if self.embedding_model == "openai":
                import openai
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                return response['data'][0]['embedding']
            
            # 如果没有嵌入模型，返回默认向量
            # 注意：在实际应用中，应该使用适当的嵌入模型
            # 这里仅作为占位符
            return [0.0] * 1536
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * 1536
    
    def __call__(self, query: str, limit: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """使用混合搜索检索知识
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            alpha: 向量搜索权重 (0-1)，BM25权重为1-alpha
            
        Returns:
            搜索结果列表
        """
        # 验证输入参数
        if not query or not isinstance(query, str):
            logger.error("Invalid query parameter")
            return []
        
        # 确保limit是有效的正整数
        limit = max(1, min(100, int(limit)))
        
        # 确保alpha在有效范围内
        alpha = max(0.0, min(1.0, float(alpha)))
        
        try:
            # 检查knowledge_manager是否初始化成功
            if self.knowledge_manager is None:
                logger.error("Knowledge manager is not initialized")
                return []
            
            # 尝试执行混合搜索，但考虑到可能的错误
            try:
                # 生成查询嵌入向量
                query_embedding = self._generate_embedding(query)
                
                # 执行混合搜索
                results = self.knowledge_manager.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    limit=limit,
                    alpha=alpha
                )
                
                # 格式化结果
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "topic": result.get("topic", ""),
                        "content": result.get("content", ""),
                        "source": result.get("source", ""),
                        "score": result.get("score", 0)
                    })
                
                logger.info("Hybrid search completed successfully")
                return formatted_results
            except Exception as hybrid_error:
                logger.warning(f"Hybrid search failed, falling back to BM25: {hybrid_error}")
                
                # 如果混合搜索失败，回退到BM25搜索
                try:
                    # 尝试使用BM25搜索
                    bm25_results = self.knowledge_manager.search_bm25(query, limit)
                    # 获取详细信息
                    detailed_results = []
                    for doc_id, score in bm25_results:
                        # 查询详细信息
                        try:
                            knowledge_item = self.knowledge_manager.get_knowledge_by_id(doc_id)
                            if knowledge_item:
                                detailed_results.append({
                                    "id": doc_id,
                                    "topic": knowledge_item.get("topic", ""),
                                    "content": knowledge_item.get("content", ""),
                                    "source": knowledge_item.get("source", ""),
                                    "created_at": knowledge_item.get("created_at", ""),
                                    "score": score
                                })
                        except Exception as item_error:
                            logger.error(f"Error retrieving knowledge item {doc_id}: {item_error}")
                    
                    logger.info("BM25 search completed successfully")
                    return detailed_results
                except Exception as bm25_error:
                    logger.warning(f"BM25 search failed, falling back to simple search: {bm25_error}")
                    
                    # 最后回退到简单的SQL查询
                    try:
                        import sqlite3
                        # 假设KnowledgeManager有一个db_path属性
                        db_path = getattr(self.knowledge_manager, 'db_path', ':memory:')
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        
                        cursor.execute('''
                        SELECT id, topic, content, source, created_at 
                        FROM knowledge_items 
                        WHERE topic LIKE ? OR content LIKE ? 
                        LIMIT ?
                        ''', (f"%{query}%", f"%{query}%", limit))
                        
                        results = cursor.fetchall()
                        conn.close()
                        
                        simple_results = [{
                            "id": result[0],
                            "topic": result[1],
                            "content": result[2],
                            "source": result[3],
                            "created_at": result[4],
                            "score": 1.0
                        } for result in results]
                        
                        logger.info("Simple SQL search completed successfully")
                        return simple_results
                    except Exception as simple_error:
                        logger.error(f"Simple SQL search failed: {simple_error}")
                        return []
        except Exception as e:
            logger.error(f"Unexpected error during knowledge retrieval: {e}")
            return []

class ReportResultTool:
    name = "report_result"
    description = "Report the final result of the troubleshooting process."
    
    def __call__(self, summary: str, result: str) -> str:
        """Report the final result."""
        return f"Report generated successfully.\nSummary: {summary}\nResult: {result}"

# Create tool instances
memory = SQLiteMemory()

list_log_files_tool = ListLogFilesTool()
read_error_logs_tool = ReadErrorLogsTool()
write_ps1_file_tool = WritePS1FileTool()
run_ps1_test_tool = RunPS1TestTool()
online_search_tool = OnlineSearchTool()
knowledge_retrieval_tool = KnowledgeRetrievalTool()  # 不再需要传入memory参数
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
