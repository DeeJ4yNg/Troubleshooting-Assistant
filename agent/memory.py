import sqlite3
import json
from datetime import datetime

class SQLiteMemory:
    def __init__(self, db_path="agent_memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create plans table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT UNIQUE,
            user_query TEXT,
            plan_content TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        ''')
        
        # Create plan_tasks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS plan_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT,
            task_id TEXT,
            description TEXT,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (plan_id) REFERENCES plans(plan_id)
        )
        ''')
        
        # Create conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP
        )
        ''')
        
        # Create knowledge table for RAG
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            content TEXT,
            source TEXT,
            created_at TIMESTAMP
        )
        ''')
        
        # Create observations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT,
            task_id TEXT,
            observation TEXT,
            success BOOLEAN,
            details TEXT,
            llm_analysis TEXT,
            should_modify_plan BOOLEAN,
            plan_modifications TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (plan_id) REFERENCES plans(plan_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def clear_memory(self):
        """Clear all memory tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tables = ["plans", "plan_tasks", "conversations", "knowledge", "observations"]
        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
            
        conn.commit()
        conn.close()

    def save_plan(self, plan_id, user_query, plan_content):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT OR REPLACE INTO plans (plan_id, user_query, plan_content, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ''', (plan_id, user_query, json.dumps(plan_content), timestamp, timestamp))
        
        conn.commit()
        conn.close()
    
    def get_plan(self, plan_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT plan_content FROM plans WHERE plan_id = ?', (plan_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
    
    def update_plan_task_status(self, plan_id, task_id, status):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE plan_tasks SET status = ? WHERE plan_id = ? AND task_id = ?
        ''', (status, plan_id, task_id))
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, conversation_id, role, content):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO conversations (conversation_id, role, content, timestamp)
        VALUES (?, ?, ?, ?)
        ''', (conversation_id, role, content, timestamp))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, conversation_id, limit=10):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT role, content, timestamp FROM conversations 
        WHERE conversation_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        ''', (conversation_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in reversed(results)]
    
    def add_knowledge(self, topic, content, source="system"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO knowledge (topic, content, source, created_at)
        VALUES (?, ?, ?, ?)
        ''', (topic, content, source, timestamp))
        
        conn.commit()
        conn.close()
    
    def retrieve_knowledge(self, query, limit=5):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT topic, content, source FROM knowledge 
        WHERE topic LIKE ? OR content LIKE ? 
        ORDER BY created_at DESC 
        LIMIT ?
        ''', (f"%{query}%", f"%{query}%", limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{"topic": r[0], "content": r[1], "source": r[2]} for r in results]
    
    def save_observation(self, plan_id, task_id, observation, success, details, llm_analysis=None, should_modify_plan=False, plan_modifications=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO observations (plan_id, task_id, observation, success, details, llm_analysis, should_modify_plan, plan_modifications, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (plan_id, task_id, observation, success, json.dumps(details), 
              json.dumps(llm_analysis) if llm_analysis else None,
              should_modify_plan,
              json.dumps(plan_modifications) if plan_modifications else None,
              timestamp))
        
        conn.commit()
        conn.close()
    
    def get_observations_for_plan(self, plan_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT task_id, observation, success, details, llm_analysis, should_modify_plan, plan_modifications, created_at 
        FROM observations 
        WHERE plan_id = ? 
        ORDER BY created_at ASC
        ''', (plan_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            "task_id": r[0],
            "observation": r[1],
            "success": r[2],
            "details": json.loads(r[3]) if r[3] else None,
            "llm_analysis": json.loads(r[4]) if r[4] else None,
            "should_modify_plan": r[5],
            "plan_modifications": json.loads(r[6]) if r[6] else None,
            "created_at": r[7]
        } for r in results]
