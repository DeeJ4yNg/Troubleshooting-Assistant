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

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ANN algo
ANN_ENGINE = "faiss" if FAISS_AVAILABLE else "linear"

class KnowledgeManager:
    def __init__(self, db_path="knowledge_db.db", embedding_dim=None, preferred_ann_engine=None):
        """init knowledge manager
        
        Args:
            db_path: knowledge db path
            embedding_dim: embedding vector dimension (default from env var, or 1536)
            preferred_ann_engine: deprecated, kept for compatibility
        """
        self.db_path = db_path
        if embedding_dim is None:
            self.embedding_dim = int(os.getenv("EMBEDDING_DIM", 1536))
        else:
            self.embedding_dim = embedding_dim
            
        self._init_db()

        self._init_ann_index()

        self._init_bm25()
    
    def _init_db(self):
        """init knowledge db table structure"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create knowledge items table
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
        
        # Create keywords table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            knowledge_id INTEGER,
            keyword TEXT,
            frequency INTEGER,
            FOREIGN KEY (knowledge_id) REFERENCES knowledge_items(id) ON DELETE CASCADE
        )
        ''')
        
        # Create full-text search virtual table
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
            topic, 
            content,
            content=knowledge_items,
            content_rowid=id
        )
        ''')
        
        # Create triggers to keep FTS table updated
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
        """init ann index"""
        try:
            self.id_map = {} # Map FAISS internal IDs to database IDs
            self.reverse_id_map = {} # Map database IDs to FAISS internal IDs
            
            if ANN_ENGINE == "faiss":
                # Create FAISS index
                # Use IndexFlatL2 for exact search or IndexIVFFlat for faster search on large datasets
                # Use IndexFlatL2 for simplicity and accuracy on smaller datasets
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            elif ANN_ENGINE == "linear":
                # Linear search doesn't need pre-built index
                self.index = None
                self.linear_embeddings = []
                self.linear_ids = []
            
            # Rebuild index from existing data
            self._rebuild_ann_index()
        except Exception as e:
            print(f"Warning: Failed to initialize {ANN_ENGINE} index: {e}")
            self.index = None
    
    def _init_bm25(self):
        """init bm25 index"""

        self.k1 = 1.5
        self.b = 0.75
        # Update BM25 statistics from existing knowledge items
        self._update_bm25_stats()
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                   splitter_type: str = "recursive", token_based: bool = False) -> List[str]:
        """Chunk text using langchain
        
        Args:
            text: Text to be chunked
            chunk_size: Maximum length of each chunk
            chunk_overlap: Overlap length between chunks
            splitter_type: Type of splitter ('recursive', 'character')
            token_based: Whether to use token-based splitting (only effective when langchain is available)
            
        Returns:
            List of chunked text
        """
        # If text is shorter than chunk size, return original text
        if len(text) <= chunk_size:
            return [text]
        
        # If langchain is available, use langchain's text splitter
        if LANGCHAIN_AVAILABLE:
            try:
                if token_based:
                    # Use token-based splitter
                    splitter = TokenTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                elif splitter_type == "character":
                    # Use character-based splitter
                    splitter = CharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separator="\n\n"
                    )
                else:
                    # Default to recursive character splitter
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=["\n\n", "\n", " ", ""]
                    )
                
                return splitter.split_text(text)
            except Exception as e:
                print(f"Warning: Langchain text splitting failed: {e}")
                # Fallback to simple chunking if langchain fails
        
        # Simple chunking fallback strategy
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # If end is not at the end of text, try to find a natural break point
            if end < len(text):
                # Find the last punctuation mark before the end of chunk
                last_punctuation = max(
                    text.rfind('.', start, end),
                    text.rfind('?', start, end),
                    text.rfind('!', start, end)
                )
                if last_punctuation > start + chunk_size * 0.8:  # If punctuation is in the last 20% of chunk, move end to punctuation
                    end = last_punctuation + 1
            chunks.append(text[start:end].strip())
            start = end - chunk_overlap
        
        return chunks
    
    def _rebuild_ann_index(self):
        """Rebuild ANN index from existing knowledge items"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Fetch all knowledge items with embeddings
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
                        
                        # Check dimension
                        if len(embedding) != self.embedding_dim:
                            print(f"Warning: Skipping item {item_id} with embedding dimension {len(embedding)} (expected {self.embedding_dim})")
                            continue

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
                        
                        # Check dimension
                        if len(embedding) != self.embedding_dim:
                            print(f"Warning: Skipping item {item_id} with embedding dimension {len(embedding)} (expected {self.embedding_dim})")
                            continue

                        self.linear_embeddings.append(embedding)
                        self.linear_ids.append(item_id)
                
                if self.linear_embeddings:
                    self.linear_embeddings = np.array(self.linear_embeddings)
                    
        except Exception as e:
            print(f"Warning: Failed to rebuild {ANN_ENGINE} index: {e}")
    
    def _update_bm25_stats(self):
        """Update BM25 statistics from existing knowledge items"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fetch all documents
        cursor.execute('SELECT id, content FROM knowledge_items')
        docs = cursor.fetchall()
        
        conn.close()
        
        if not docs:
            self.avg_doc_len = 0
            self.doc_count = 0
            self.word_freq = {}
            return
        
        self.doc_count = len(docs)
        
        # Calculate average document length and word frequency
        total_len = 0
        self.word_freq = {}
        
        for doc_id, content in docs:
            words = self._tokenize(content)
            total_len += len(words)
            
            # Update word frequency
            doc_word_set = set(words)
            for word in doc_word_set:
                if word not in self.word_freq:
                    self.word_freq[word] = 0
                self.word_freq[word] += 1
        
        self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, removing punctuation and stop words"""
        # Convert to lowercase and remove non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Tokenize and remove stop words (simple implementation)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in text.split() if word not in stop_words and len(word) > 1]
        return words
    
    def add_knowledge(self, topic: str, content: str, source: str = "system", embedding: Optional[List[float]] = None, 
                     auto_chunk: bool = False, chunk_size: int = 1000, chunk_overlap: int = 200, 
                     chunk_splitter_type: str = "recursive", token_based_chunking: bool = False) -> List[int]:
        """Add knowledge item, supporting automatic chunking of large documents
        
        Args:
            topic: main topic of the knowledge item
            content: content of the knowledge item
            source: source of the knowledge item
            embedding: optional embedding vector (if provided, will be used for all chunks)
            auto_chunk: whether to automatically chunk large documents
            chunk_size: chunk size for automatic chunking
            chunk_overlap: overlap size between chunks
            chunk_splitter_type: type of chunk splitter ('recursive', 'character')
            token_based_chunking: whether to chunk based on token count (True) or character count (False)
            
        Returns:
            list of IDs of the added knowledge items
        """
        # Check if chunking is needed
        if auto_chunk and len(content) > chunk_size:
            # Chunk the content
            chunks = self.chunk_text(
                content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                splitter_type=chunk_splitter_type,
                token_based=token_based_chunking
            )
            
            # Add each chunk as a separate knowledge item
            knowledge_ids = []
            for i, chunk in enumerate(chunks):
                # Generate unique sub-topic for each chunk
                chunk_topic = f"{topic} - Part {i+1}/{len(chunks)}"
                # Add chunk content
                chunk_id = self._add_single_knowledge_item(chunk_topic, chunk, source, embedding)
                knowledge_ids.append(chunk_id)
            
            return knowledge_ids
        else:
            # Add the full content as a single knowledge item
            knowledge_id = self._add_single_knowledge_item(topic, content, source, embedding)
            return [knowledge_id]
    
    def _add_single_knowledge_item(self, topic: str, content: str, source: str, embedding: Optional[List[float]] = None) -> int:
        """Add a single knowledge item to the database
        
        Args:
            topic: Topic of the knowledge item
            content: Content of the knowledge item
            source: Source of the knowledge item
            embedding: Optional embedding vector
            
        Returns:
            ID of the newly added knowledge item
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Process embedding vector
        embedding_blob = None
        if embedding:
            # Convert to numpy array and serialize to binary
            embedding_array = np.array(embedding, dtype=np.float32)
            embedding_blob = embedding_array.tobytes()
        
        # Insert knowledge item
        cursor.execute('''
        INSERT INTO knowledge_items (topic, content, source, embedding, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (topic, content, source, embedding_blob, timestamp, timestamp))
        
        knowledge_id = cursor.lastrowid
        
        # Update keywords table
        self._update_keywords(cursor, knowledge_id, content)
        
        conn.commit()
        conn.close()
        
        # Update index if embedding is provided
        if embedding:
            self._rebuild_ann_index()
        self._update_bm25_stats()
        
        return knowledge_id
    
    def update_knowledge(self, knowledge_id: int, topic: Optional[str] = None, 
                        content: Optional[str] = None, embedding: Optional[List[float]] = None) -> bool:
        """Update a knowledge item
        
        Args:
            knowledge_id: ID of the knowledge item to update
            topic: New topic (optional)
            content: New content (optional)
            embedding: New embedding vector (optional)
            
        Returns:
            Whether the update was successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if knowledge item exists
        cursor.execute('SELECT id FROM knowledge_items WHERE id = ?', (knowledge_id,))
        if not cursor.fetchone():
            conn.close()
            return False
        
        timestamp = datetime.now().isoformat()
        
        # Build update statement
        updates = []
        params = []
        
        if topic is not None:
            updates.append("topic = ?")
            params.append(topic)
        
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        
        if embedding is not None:
            # Convert to numpy array and serialize to binary
            embedding_array = np.array(embedding, dtype=np.float32)
            embedding_blob = embedding_array.tobytes()
            updates.append("embedding = ?")
            params.append(embedding_blob)
        
        updates.append("updated_at = ?")
        params.append(timestamp)
        params.append(knowledge_id)
        
        # Execute update
        query = f"UPDATE knowledge_items SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        
        # Update keywords table if content changed
        if content is not None:
            # Delete old keywords
            cursor.execute('DELETE FROM keywords WHERE knowledge_id = ?', (knowledge_id,))
            # Add new keywords
            self._update_keywords(cursor, knowledge_id, content)
        
        conn.commit()
        conn.close()
        
        # Update index if embedding or content changed
        if embedding is not None or content is not None:
            self._rebuild_ann_index()
            self._update_bm25_stats()
        
        return True
    
    def delete_knowledge(self, knowledge_id: int) -> bool:
        """Delete a knowledge item
        
        Args:
            knowledge_id: ID of the knowledge item to delete
            
        Returns:
            Whether the deletion was successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if knowledge item exists
        cursor.execute('SELECT id FROM knowledge_items WHERE id = ?', (knowledge_id,))
        if not cursor.fetchone():
            conn.close()
            return False
        
        # Delete knowledge item (keywords will be cascaded)
        cursor.execute('DELETE FROM knowledge_items WHERE id = ?', (knowledge_id,))
        
        conn.commit()
        conn.close()
        
        # Rebuild index
        self._rebuild_ann_index()
        self._update_bm25_stats()
        
        return True
    
    def _update_keywords(self, cursor: sqlite3.Cursor, knowledge_id: int, content: str):
        """Update keywords table for a knowledge item
        
        Args:
            cursor: Database cursor
            knowledge_id: ID of the knowledge item
            content: Content of the knowledge item
        """
        words = self._tokenize(content)
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
        
        # Insert keywords
        for word, freq in word_counts.items():
            cursor.execute('''
            INSERT INTO keywords (knowledge_id, keyword, frequency)
            VALUES (?, ?, ?)
            ''', (knowledge_id, word, freq))
    
    def get_knowledge_by_id(self, knowledge_id: int) -> Optional[Dict[str, Any]]:
        """Get a knowledge item by ID
        
        Args:
            knowledge_id: ID of the knowledge item to retrieve
            
        Returns:
            Dictionary containing knowledge item details or None if not found
        """
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
        """Search knowledge items using BM25 algorithm
        
        Args:
            query: Query text
            limit: Maximum number of results to return
            
        Returns:
            List of tuples containing (knowledge_id, score)
        """
        if self.doc_count == 0:
            return []
        
        # Tokenize query
        query_words = self._tokenize(query)
        if not query_words:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fetch all documents
        cursor.execute('SELECT id, content FROM knowledge_items')
        docs = cursor.fetchall()
        
        scores = []
        
        for doc_id, content in docs:
            doc_words = self._tokenize(content)
            doc_len = len(doc_words)
            
            # Calculate BM25 score
            score = 0
            for word in query_words:
                # Calculate term frequency in document
                tf = doc_words.count(word)
                # Calculate inverse document frequency
                df = self.word_freq.get(word, 0)
                idf = np.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
                # Calculate BM25 term
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
            
            scores.append((doc_id, score))
        
        conn.close()
        
        # Sort by score and return top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]
    
    def search_vector(self, query_embedding: List[float], limit: int = 5) -> List[Tuple[int, float]]:
        """Search knowledge items using vector similarity search
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results to return
            
        Returns:
            List of tuples containing (knowledge_id, similarity_score)
        """
        try:
            # Convert query vector to numpy array
            query_vec = np.array(query_embedding, dtype=np.float32)
            
            # Search using FAISS if available
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
                # Linear search
                if not self.linear_embeddings.size:
                    return []
                
                # Calculate cosine similarity with all vectors
                # Ensure vectors are normalized to use dot product as cosine similarity
                # Simple implementation: use dot product approximation of cosine similarity (if vectors are normalized)
                similarities = []
                for i, embedding in enumerate(self.linear_embeddings):
                    # Calculate cosine similarity
                    dot_product = np.dot(query_vec, embedding)
                    norm_product = np.linalg.norm(query_vec) * np.linalg.norm(embedding)
                    if norm_product > 0:
                        similarity = dot_product / norm_product
                    else:
                        similarity = 0
                    similarities.append((self.linear_ids[i], similarity))
                
                # Sort by similarity score in descending order
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Return top N results
                return similarities[:limit]
                
        except Exception as e:
            print(f"Warning: Vector search failed with {ANN_ENGINE}: {e}")
            
        # Return empty list if all methods fail
        return []
    
    def hybrid_search(self, query: str, query_embedding: Optional[List[float]] = None, 
                     limit: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Hybrid search combining BM25 and vector similarity search
        
        Args:
            query: Query text
            query_embedding: Query vector (optional)
            limit: Maximum number of results to return
            alpha: Weight for vector similarity search (0-1), BM25 weight is 1-alpha
            
        Returns:
            List of search results with combined scores
        """
        try:
            # Execute BM25 search
            bm25_results = self.search_bm25(query, limit * 2)  # Get more results for merging
            
            # Execute vector search (if embedding is provided and ANN engine is available)
            vector_results = []
            if ANN_ENGINE != "linear" and query_embedding:
                vector_results = self.search_vector(query_embedding, limit * 2)
            
            # Merge results
            combined_scores = {}
            
            # Normalize BM25 scores (if any results)
            if bm25_results:
                max_bm25_score = max(score for _, score in bm25_results)
                if max_bm25_score > 0:
                    for doc_id, score in bm25_results:
                        combined_scores[doc_id] = (1 - alpha) * (score / max_bm25_score)
                else:
                    # if all BM25 scores are 0, use equal scores
                    for doc_id, _ in bm25_results:
                        combined_scores[doc_id] = 1.0
            
            # Normalize vector scores and merge (if any results)
            if vector_results:
                # Find max vector score for normalization
                max_vector_score = max(score for _, score in vector_results)
                if max_vector_score > 0:
                    for doc_id, score in vector_results:
                        normalized_score = score / max_vector_score
                        if doc_id in combined_scores:
                            combined_scores[doc_id] += alpha * normalized_score
                        else:
                            combined_scores[doc_id] = alpha * normalized_score
                else:
                    # if all vector scores are 0, still add these documents but give lower weight
                    for doc_id, _ in vector_results:
                        if doc_id in combined_scores:
                            combined_scores[doc_id] += alpha * 0.1  # give lower weight
                        else:
                            combined_scores[doc_id] = alpha * 0.1
            
            # If only BM25 results, use them directly
            if not vector_results and bm25_results:
                combined_scores = {doc_id: score for doc_id, score in bm25_results}
            
            # If no results from both BM25 and vector, try SQLite FTS5
            if not combined_scores:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Use FTS5 for full-text search
                cursor.execute('''
                SELECT rowid FROM knowledge_fts 
                WHERE knowledge_fts MATCH ? 
                LIMIT ?
                ''', (query, limit))
                
                fts_results = cursor.fetchall()
                conn.close()
                
                # Assign default score to FTS results
                for row in fts_results:
                    combined_scores[row[0]] = 1.0
            
            # Sort by combined score in descending order
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get detailed information
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
            # Fallback to simple SQL query if hybrid search fails
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
        """List all knowledge items
        
        Args:
            limit: Number of items to return
            
        Returns:
            List of knowledge items
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
