from agent import knowledge_manager
from agent.knowledge_manager import KnowledgeManager
from typing import List, Optional
import os
from dotenv import load_dotenv
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Load environment variables
load_dotenv()

def generate_embedding(text: str, model_type: Optional[str] = None) -> List[float]:
    """generate embeddings
    
    Args:
        text: input txt
        model_type: model ("openai" / None)
        
    Returns:
        embedding list
    """
    #Use Langchain to generate embedding
    try:
        from langchain.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_EMBEDDING_MODEL"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        return embeddings.embed_query(text)
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return [0.0] * 1536

def Add_Knowledge(file_path: str, topic: str = "General Knowledge") -> str:
    """Read file and add to knowledge base."""
    knowledge_manager = KnowledgeManager()
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist"
        
    ext = os.path.splitext(file_path)[1].lower()
    content = ""
        
    try:
        if ext == ".docx":
            if not DOCX_AVAILABLE:
                return "Error: python-docx module not available. Cannot read .docx files."
            doc = docx.Document(file_path)
            content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        elif ext in [".txt", ".md"]:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        else:
            return f"Error: Unsupported file extension '{ext}'"
            
        if not content.strip():
            return f"Error: File '{file_path}' is empty"
            
        # Chunking logic
        chunk_size = 1000
        chunk_overlap = 200
            
        # Simple chunking
        chunks = []
        start = 0
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            start += chunk_size - chunk_overlap
            
        # Add chunks to knowledge base
        added_count = 0
        for i, chunk in enumerate(chunks):
            chunk_topic = f"{topic} - Part {i+1}/{len(chunks)}"
            embedding = generate_embedding(chunk)
                
            knowledge_manager.add_knowledge(
                topic=chunk_topic,
                content=chunk,
                source=os.path.basename(file_path),
                embedding=embedding
            )
            added_count += 1
                
        return f"Successfully ingested '{file_path}' into knowledge base. Added {added_count} chunks."
            
    except Exception as e:
        return f"Error processing file: {str(e)}"

Add_Knowledge("CannotCreateExcelFile.docx")
