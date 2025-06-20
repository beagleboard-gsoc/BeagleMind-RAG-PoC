from langchain_groq import ChatGroq
from .config import GROQ_API_KEY, GROQ_MODEL_NAME

class GraphQASystem:
    def __init__(self, retrieval_system):
        self.retrieval_system = retrieval_system
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL_NAME,
            temperature=0.1
        )
    
    def ask_question(self, question):
        """Ask a question and get an answer with sources"""
        # Get relevant documents from Milvus
        search_results = self.retrieval_system.search(question, top_k=5)
        
        if not search_results:
            return {"answer": "No relevant documents found.", "sources": []}
        
        # Combine context from search results
        context = "\n\n".join([text for text, score in search_results])
        
        # Create prompt
        prompt = f"""Use the following context to answer the question. If you cannot answer based on the context, say so.

Context: {context}

Question: {question}

Answer:"""
        
        # Get answer from Groq
        response = self.llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "sources": [{"text": text, "score": score} for text, score in search_results]
        }
