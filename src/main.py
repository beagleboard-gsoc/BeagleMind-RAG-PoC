import os
from . import config
from .retrieval import RetrievalSystem
from .qa_system import QASystem

def setup_system():
    """Initialize and setup the RAG system"""
    # Initialize retrieval system
    retrieval_system = RetrievalSystem()
    
    # Create collection
    print("Creating Milvus collection...")
    
    
    retrieval_system.create_collection()
    
    # Initialize QA system
    qa_system = QASystem(retrieval_system)
    
    return retrieval_system, qa_system

def generate_answer(question, qa_system):
    """Generate answer for a given question"""
    try:
        result = qa_system.ask_question(question)
        return result["answer"]
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Setup the system
    retrieval_system, qa_system = setup_system()
    
    if qa_system is None:
        print("Failed to setup system. Exiting.")
        exit(1)
    
    print("\nRAG System ready! Type 'exit' to quit.")
    
    while True:
        q = input("\nYou: ")
        if q.strip().lower() == "exit":
            break
        
        answer = generate_answer(q, qa_system)
        print(f"Bot: {answer}")
