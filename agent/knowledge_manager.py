import sqlite3
import json
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# 尝试导入langchain相关模块用于文档分块
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain not available. Text chunking functionality will be limited.")

# 尝试导入faiss
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not available. Using linear search as fallback.")

# 设置当前使用的ANN算法
ANN_ENGINE = "faiss" if FAISS_AVAILABLE else "linear"

class KnowledgeManager:
    def __init__(self, db_path="knowledge_db.db", embedding_dim=1536, preferred_ann_engine=None):
        """初始化知识库管理器
        
        Args:
            db_path: 数据库文件路径
            embedding_dim: 嵌入向量维度（默认1536，适合OpenAI等模型）
            preferred_ann_engine: 已弃用，保留是为了兼容性
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._init_db()
        
        # 初始化ANN索引
        self._init_ann_index()
        
        # 初始化BM25相关数据
        self._init_bm25()
    
    def _init_db(self):
        """初始化SQLite数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建知识表，包含向量字段
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            content TEXT,
            source TEXT,
            embedding BLOB,  -- 存储向量的二进制数据
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        ''')
        
        # 创建关键词索引表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            knowledge_id INTEGER,
            keyword TEXT,
            frequency INTEGER,
            FOREIGN KEY (knowledge_id) REFERENCES knowledge_items(id) ON DELETE CASCADE
        )
        ''')
        
        # 创建全文搜索虚拟表（使用FTS5）
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
            topic, 
            content,
            content=knowledge_items,
            content_rowid=id
        )
        ''')
        
        # 创建触发器以保持FTS表更新
        cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge_items BEGIN
            INSERT INTO knowledge_fts(rowid, topic, content) 
            VALUES (new.id, new.topic, new.content);
        END
        ''')
        
        cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge_items BEGIN
            DELETE FROM knowledge_fts WHERE rowid = old.id;
        END
        ''')
        
        cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge_items BEGIN
            UPDATE knowledge_fts SET topic = new.topic, content = new.content 
            WHERE rowid = new.id;
        END
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_ann_index(self):
        """初始化ANN索引"""
        try:
            self.id_map = {} # Map FAISS internal IDs to database IDs
            self.reverse_id_map = {} # Map database IDs to FAISS internal IDs
            
            if ANN_ENGINE == "faiss":
                # Create FAISS index
                # We use IndexFlatL2 for exact search or IndexIVFFlat for faster search on large datasets
                # Here we use IndexFlatL2 for simplicity and accuracy on smaller datasets
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            elif ANN_ENGINE == "linear":
                # Linear search doesn't need pre-built index
                self.index = None
                self.linear_embeddings = []
                self.linear_ids = []
            
            # 加载现有数据到索引
            self._rebuild_ann_index()
        except Exception as e:
            print(f"Warning: Failed to initialize {ANN_ENGINE} index: {e}")
            self.index = None
    
    def _init_bm25(self):
        """初始化BM25相关参数"""
        # BM25参数
        self.k1 = 1.5
        self.b = 0.75
        # 计算语料库统计信息
        self._update_bm25_stats()
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                   splitter_type: str = "recursive", token_based: bool = False) -> List[str]:
        """使用langchain对文本进行分块
        
        Args:
            text: 要分块的文本
            chunk_size: 每块的最大长度
            chunk_overlap: 块之间的重叠长度
            splitter_type: 分块器类型 ('recursive', 'character')
            token_based: 是否基于token计数（仅在langchain可用时有效）
            
        Returns:
            分块后的文本列表
        """
        # 如果文本较短，直接返回原文
        if len(text) <= chunk_size:
            return [text]
        
        # 如果langchain可用，使用langchain的分块器
        if LANGCHAIN_AVAILABLE:
            try:
                if token_based:
                    # 使用基于token的分块器
                    splitter = TokenTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                elif splitter_type == "character":
                    # 使用基于字符的分块器
                    splitter = CharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separator="\n\n"
                    )
                else:
                    # 默认使用递归字符分块器
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=["\n\n", "\n", " ", ""]
                    )
                
                return splitter.split_text(text)
            except Exception as e:
                print(f"Warning: Langchain text splitting failed: {e}")
                # 失败时回退到简单分块
        
        # 简单分块回退策略
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # 尝试在句子边界处分割
            if end < len(text):
                # 寻找最近的句号、问号或感叹号
                last_punctuation = max(
                    text.rfind('.', start, end),
                    text.rfind('?', start, end),
                    text.rfind('!', start, end)
                )
                if last_punctuation > start + chunk_size * 0.8:  # 确保至少80%的块大小
                    end = last_punctuation + 1
            chunks.append(text[start:end].strip())
            start = end - chunk_overlap
        
        return chunks
    
    def _rebuild_ann_index(self):
        """重建ANN索引"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取所有知识项及其嵌入向量
            cursor.execute('SELECT id, embedding FROM knowledge_items WHERE embedding IS NOT NULL')
            results = cursor.fetchall()
            
            conn.close()
            
            # Clear existing maps
            self.id_map = {}
            self.reverse_id_map = {}
            
            if ANN_ENGINE == "faiss" and self.index is not None:
                # Reset index
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                
                if not results:
                    return
                
                embeddings = []
                ids = []
                
                for i, (item_id, embedding_blob) in enumerate(results):
                    if embedding_blob:
                        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                        embeddings.append(embedding)
                        # FAISS uses integer IDs starting from 0 (if using add)
                        # or we can use IndexIDMap to supply custom IDs, but let's stick to simple map
                        self.id_map[i] = item_id
                        self.reverse_id_map[item_id] = i
                
                if embeddings:
                    embeddings_np = np.array(embeddings).astype('float32')
                    self.index.add(embeddings_np)
                    
            elif ANN_ENGINE == "linear":
                # Prepare data for linear search
                self.linear_embeddings = []
                self.linear_ids = []
                
                for item_id, embedding_blob in results:
                    if embedding_blob:
                        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                        self.linear_embeddings.append(embedding)
                        self.linear_ids.append(item_id)
                
                if self.linear_embeddings:
                    self.linear_embeddings = np.array(self.linear_embeddings)
                    
        except Exception as e:
            print(f"Warning: Failed to rebuild {ANN_ENGINE} index: {e}")
    
    def _update_bm25_stats(self):
        """更新BM25统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取所有文档
        cursor.execute('SELECT id, content FROM knowledge_items')
        docs = cursor.fetchall()
        
        conn.close()
        
        if not docs:
            self.avg_doc_len = 0
            self.doc_count = 0
            self.word_freq = {}
            return
        
        self.doc_count = len(docs)
        
        # 计算平均文档长度和词频
        total_len = 0
        self.word_freq = {}
        
        for doc_id, content in docs:
            words = self._tokenize(content)
            total_len += len(words)
            
            # 统计词频
            doc_word_set = set(words)
            for word in doc_word_set:
                if word not in self.word_freq:
                    self.word_freq[word] = 0
                self.word_freq[word] += 1
        
        self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 0
    
    def _tokenize(self, text: str) -> List[str]:
        """将文本分词"""
        # 转为小写并移除非字母数字字符
        text = re.sub(r'[^\w\s]', '', text.lower())
        # 分词并移除停用词（简单实现）
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in text.split() if word not in stop_words and len(word) > 1]
        return words
    
    def add_knowledge(self, topic: str, content: str, source: str = "system", embedding: Optional[List[float]] = None, 
                     auto_chunk: bool = False, chunk_size: int = 1000, chunk_overlap: int = 200, 
                     chunk_splitter_type: str = "recursive", token_based_chunking: bool = False) -> List[int]:
        """添加知识项，支持自动分块大文档
        
        Args:
            topic: 主题
            content: 内容
            source: 来源
            embedding: 可选的嵌入向量（如果提供，将用于所有块）
            auto_chunk: 是否自动分块大文档
            chunk_size: 分块大小
            chunk_overlap: 块之间的重叠大小
            chunk_splitter_type: 分块器类型 ('recursive', 'character')
            token_based_chunking: 是否基于token计数分块
            
        Returns:
            添加的知识项ID列表
        """
        # 检查是否需要分块
        if auto_chunk and len(content) > chunk_size:
            # 对内容进行分块
            chunks = self.chunk_text(
                content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                splitter_type=chunk_splitter_type,
                token_based=token_based_chunking
            )
            
            # 为每个块添加知识项
            knowledge_ids = []
            for i, chunk in enumerate(chunks):
                # 为每个块生成唯一的子主题
                chunk_topic = f"{topic} - Part {i+1}/{len(chunks)}"
                # 添加块内容
                chunk_id = self._add_single_knowledge_item(chunk_topic, chunk, source, embedding)
                knowledge_ids.append(chunk_id)
            
            return knowledge_ids
        else:
            # 不进行分块，直接添加完整内容
            knowledge_id = self._add_single_knowledge_item(topic, content, source, embedding)
            return [knowledge_id]
    
    def _add_single_knowledge_item(self, topic: str, content: str, source: str, embedding: Optional[List[float]] = None) -> int:
        """添加单个知识项
        
        Args:
            topic: 主题
            content: 内容
            source: 来源
            embedding: 可选的嵌入向量
            
        Returns:
            新添加知识项的ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # 处理嵌入向量
        embedding_blob = None
        if embedding:
            # 转换为numpy数组并序列化为二进制
            embedding_array = np.array(embedding, dtype=np.float32)
            embedding_blob = embedding_array.tobytes()
        
        # 插入知识项
        cursor.execute('''
        INSERT INTO knowledge_items (topic, content, source, embedding, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (topic, content, source, embedding_blob, timestamp, timestamp))
        
        knowledge_id = cursor.lastrowid
        
        # 更新关键词表
        self._update_keywords(cursor, knowledge_id, content)
        
        conn.commit()
        conn.close()
        
        # 更新索引
        if embedding:
            self._rebuild_ann_index()
        self._update_bm25_stats()
        
        return knowledge_id
    
    def update_knowledge(self, knowledge_id: int, topic: Optional[str] = None, 
                        content: Optional[str] = None, embedding: Optional[List[float]] = None) -> bool:
        """更新知识项
        
        Args:
            knowledge_id: 知识项ID
            topic: 新主题（可选）
            content: 新内容（可选）
            embedding: 新嵌入向量（可选）
            
        Returns:
            更新是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查知识项是否存在
        cursor.execute('SELECT id FROM knowledge_items WHERE id = ?', (knowledge_id,))
        if not cursor.fetchone():
            conn.close()
            return False
        
        timestamp = datetime.now().isoformat()
        
        # 构建更新语句
        updates = []
        params = []
        
        if topic is not None:
            updates.append("topic = ?")
            params.append(topic)
        
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        
        if embedding is not None:
            # 转换为numpy数组并序列化为二进制
            embedding_array = np.array(embedding, dtype=np.float32)
            embedding_blob = embedding_array.tobytes()
            updates.append("embedding = ?")
            params.append(embedding_blob)
        
        updates.append("updated_at = ?")
        params.append(timestamp)
        params.append(knowledge_id)
        
        # 执行更新
        query = f"UPDATE knowledge_items SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        
        # 如果内容更新了，更新关键词表
        if content is not None:
            # 删除旧关键词
            cursor.execute('DELETE FROM keywords WHERE knowledge_id = ?', (knowledge_id,))
            # 添加新关键词
            self._update_keywords(cursor, knowledge_id, content)
        
        conn.commit()
        conn.close()
        
        # 更新索引
        if embedding is not None or content is not None:
            self._rebuild_ann_index()
            self._update_bm25_stats()
        
        return True
    
    def delete_knowledge(self, knowledge_id: int) -> bool:
        """删除知识项
        
        Args:
            knowledge_id: 知识项ID
            
        Returns:
            删除是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查知识项是否存在
        cursor.execute('SELECT id FROM knowledge_items WHERE id = ?', (knowledge_id,))
        if not cursor.fetchone():
            conn.close()
            return False
        
        # 删除知识项（关键词会通过外键级联删除）
        cursor.execute('DELETE FROM knowledge_items WHERE id = ?', (knowledge_id,))
        
        conn.commit()
        conn.close()
        
        # 重建索引
        self._rebuild_ann_index()
        self._update_bm25_stats()
        
        return True
    
    def _update_keywords(self, cursor: sqlite3.Cursor, knowledge_id: int, content: str):
        """更新关键词表"""
        words = self._tokenize(content)
        
        # 统计词频
        word_counts = {}
        for word in words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
        
        # 插入关键词
        for word, freq in word_counts.items():
            cursor.execute('''
            INSERT INTO keywords (knowledge_id, keyword, frequency)
            VALUES (?, ?, ?)
            ''', (knowledge_id, word, freq))
    
    def get_knowledge_by_id(self, knowledge_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取知识项"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, topic, content, source, created_at, updated_at 
        FROM knowledge_items 
        WHERE id = ?
        ''', (knowledge_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "id": result[0],
                "topic": result[1],
                "content": result[2],
                "source": result[3],
                "created_at": result[4],
                "updated_at": result[5]
            }
        return None
    
    def search_bm25(self, query: str, limit: int = 5) -> List[Tuple[int, float]]:
        """使用BM25算法搜索
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            
        Returns:
            [(知识ID, 分数)] 的列表
        """
        if self.doc_count == 0:
            return []
        
        # 分词查询
        query_words = self._tokenize(query)
        if not query_words:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取所有文档
        cursor.execute('SELECT id, content FROM knowledge_items')
        docs = cursor.fetchall()
        
        scores = []
        
        for doc_id, content in docs:
            doc_words = self._tokenize(content)
            doc_len = len(doc_words)
            
            # 计算BM25分数
            score = 0
            for word in query_words:
                # 计算词在文档中的频率
                tf = doc_words.count(word)
                # 计算逆文档频率
                df = self.word_freq.get(word, 0)
                idf = np.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
                # 计算BM25项
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
            
            scores.append((doc_id, score))
        
        conn.close()
        
        # 按分数排序并返回前N个
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]
    
    def search_vector(self, query_embedding: List[float], limit: int = 5) -> List[Tuple[int, float]]:
        """使用向量搜索（支持多种ANN算法）
        
        Args:
            query_embedding: 查询向量
            limit: 返回结果数量限制
            
        Returns:
            [(知识ID, 相似度)] 的列表
        """
        try:
            # 转换查询向量为numpy数组
            query_vec = np.array(query_embedding, dtype=np.float32)
            
            # 根据不同的ANN引擎执行搜索
            if ANN_ENGINE == "faiss" and self.index is not None:
                # Check if index is empty
                if self.index.ntotal == 0:
                    return []
                
                # Search
                # FAISS expects query to be (1, d) array
                D, I = self.index.search(query_vec.reshape(1, -1), limit)
                
                # Calculate similarity (convert L2 distance to similarity score if needed, or just use negative distance)
                # For L2, smaller is better. We can convert to a similarity score like 1/(1+distance)
                # or just return negative distance if the caller expects higher score = better match
                
                results = []
                for i in range(len(I[0])):
                    faiss_id = I[0][i]
                    distance = D[0][i]
                    
                    if faiss_id == -1: # No match found
                        continue
                        
                    # Get database ID
                    doc_id = self.id_map.get(faiss_id)
                    
                    if doc_id is not None:
                         # Convert L2 distance to similarity (0 to 1 range approx)
                        similarity = 1 / (1 + distance)
                        results.append((doc_id, similarity))
                
                return results
                
            elif ANN_ENGINE == "linear":
                # 线性搜索
                if not self.linear_embeddings.size:
                    return []
                
                # 计算与所有向量的余弦相似度
                # 确保向量已经归一化以使用点积作为余弦相似度
                # 简单实现：使用点积近似余弦相似度（如果向量已归一化）
                similarities = []
                for i, embedding in enumerate(self.linear_embeddings):
                    # 计算余弦相似度
                    dot_product = np.dot(query_vec, embedding)
                    norm_product = np.linalg.norm(query_vec) * np.linalg.norm(embedding)
                    if norm_product > 0:
                        similarity = dot_product / norm_product
                    else:
                        similarity = 0
                    similarities.append((self.linear_ids[i], similarity))
                
                # 按相似度降序排序
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # 返回前N个结果
                return similarities[:limit]
                
        except Exception as e:
            print(f"Warning: Vector search failed with {ANN_ENGINE}: {e}")
            
        # 所有方法都失败时返回空列表
        return []
    
    def hybrid_search(self, query: str, query_embedding: Optional[List[float]] = None, 
                     limit: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """混合搜索（BM25 + 向量搜索）
        
        Args:
            query: 查询文本
            query_embedding: 查询向量（可选）
            limit: 返回结果数量限制
            alpha: 向量搜索权重（0-1），BM25权重为1-alpha
            
        Returns:
            搜索结果列表
        """
        try:
            # 执行BM25搜索
            bm25_results = self.search_bm25(query, limit * 2)  # 获取更多结果用于合并
            
            # 执行向量搜索（如果提供了嵌入向量且ANN引擎可用）
            vector_results = []
            if ANN_ENGINE != "linear" and query_embedding:
                vector_results = self.search_vector(query_embedding, limit * 2)
            
            # 合并结果
            combined_scores = {}
            
            # 归一化BM25分数
            if bm25_results:
                max_bm25_score = max(score for _, score in bm25_results)
                if max_bm25_score > 0:
                    for doc_id, score in bm25_results:
                        combined_scores[doc_id] = (1 - alpha) * (score / max_bm25_score)
                else:
                    # 如果所有分数都为0，使用相等的分数
                    for doc_id, _ in bm25_results:
                        combined_scores[doc_id] = 1.0
            
            # 归一化向量分数并合并
            if vector_results:
                # 找到最大向量分数用于归一化
                max_vector_score = max(score for _, score in vector_results)
                if max_vector_score > 0:
                    for doc_id, score in vector_results:
                        normalized_score = score / max_vector_score
                        if doc_id in combined_scores:
                            combined_scores[doc_id] += alpha * normalized_score
                        else:
                            combined_scores[doc_id] = alpha * normalized_score
                else:
                    # 如果所有向量分数都为0，仍然添加这些文档但给予较低权重
                    for doc_id, _ in vector_results:
                        if doc_id in combined_scores:
                            combined_scores[doc_id] += alpha * 0.1  # 给予小权重
                        else:
                            combined_scores[doc_id] = alpha * 0.1
            
            # 如果只有BM25结果，直接使用
            if not vector_results and bm25_results:
                combined_scores = {doc_id: score for doc_id, score in bm25_results}
            
            # 如果没有任何结果，尝试使用SQLite的FTS5全文搜索
            if not combined_scores:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 使用FTS5进行全文搜索
                cursor.execute('''
                SELECT rowid FROM knowledge_fts 
                WHERE knowledge_fts MATCH ? 
                LIMIT ?
                ''', (query, limit))
                
                fts_results = cursor.fetchall()
                conn.close()
                
                # 为FTS结果分配默认分数
                for row in fts_results:
                    combined_scores[row[0]] = 1.0
            
            # 按分数排序
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 获取详细信息
            final_results = []
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for doc_id, score in sorted_results[:limit]:
                cursor.execute('''
                SELECT topic, content, source, created_at 
                FROM knowledge_items 
                WHERE id = ?
                ''', (doc_id,))
                result = cursor.fetchone()
                
                if result:
                    final_results.append({
                        "id": doc_id,
                        "topic": result[0],
                        "content": result[1],
                        "source": result[2],
                        "created_at": result[3],
                        "score": score
                    })
            
            conn.close()
            
            return final_results
        except Exception as e:
            print(f"Warning: Hybrid search failed: {e}")
            # 失败时回退到简单的SQL查询
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, topic, content, source, created_at 
            FROM knowledge_items 
            WHERE topic LIKE ? OR content LIKE ? 
            LIMIT ?
            ''', (f"%{query}%", f"%{query}%", limit))
            
            results = cursor.fetchall()
            conn.close()
            
            return [{
                "id": result[0],
                "topic": result[1],
                "content": result[2],
                "source": result[3],
                "created_at": result[4],
                "score": 1.0
            } for result in results]
    
    def list_all_knowledge(self, limit: int = 100) -> List[Dict[str, Any]]:
        """列出所有知识项
        
        Args:
            limit: 返回结果数量限制
            
        Returns:
            知识项列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, topic, content, source, created_at, updated_at 
        FROM knowledge_items 
        ORDER BY created_at DESC 
        LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            "id": result[0],
            "topic": result[1],
            "content": result[2],
            "source": result[3],
            "created_at": result[4],
            "updated_at": result[5]
        } for result in results]
