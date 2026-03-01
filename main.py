import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Optional RAG Context dependencies
embedding_model = None
RAG_ENABLED = os.environ.get("ENABLE_RAG", "false").lower() == "true"

try:
    from fastembed import TextEmbedding
    from pinecone import Pinecone
    if RAG_ENABLED:
        print("Loading embedding model...")
        embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"RAG dependencies failed to load: {e}")
    RAG_ENABLED = False

load_dotenv()

app = FastAPI(title="AlgoRAG API")

class ProblemRequest(BaseModel):
    prompt: str

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "groq_configured": bool(os.environ.get("GROQ_API_KEY")),
        "pinecone_configured": bool(os.environ.get("PINECONE_API_KEY")),
        "rag_enabled": RAG_ENABLED,
        "embedding_model_loaded": embedding_model is not None
    }

@app.post("/api/generate_problem")
async def generate_problem(request: ProblemRequest):
    print(f"DEBUG: Starting generation for prompt: {request.prompt[:50]}...")
    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            return {"error": True, "message": "GROQ_API_KEY is missing in environment"}
            
        rag_context = ""
        if RAG_ENABLED and embedding_model:
            try:
                print("DEBUG: Fetching RAG context...")
                pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                index = pc.Index("algoforge-rag")
                query_vector = list(embedding_model.embed([request.prompt]))[0].tolist()
                search_results = index.query(vector=query_vector, top_k=3, include_metadata=True)
                
                if search_results.get("matches"):
                    rag_context = "\n\n--- REFERENCE EXAMPLES ---\n"
                    for i, match in enumerate(search_results["matches"]):
                        meta = match.get("metadata", {})
                        rag_context += f"Example {i+1}:\nTitle: {meta.get('title')}\nDescription: {meta.get('description')}\n\n"
                        
            except Exception as rag_e:
                print(f"DEBUG RAG Error (Non-fatal): {rag_e}")
                
        system_prompt = f"""
Sana verilen konuya uygun bir algoritma problemi üret. 
Yanıtın TAMAMEN Türkçe olsun. 
Yanıtın sadece JSON formatında olsun:

{{
  "title": "Başlık",
  "description": "HTML Açıklama",
  "input_description": "Input Açıklaması",
  "output_description": "Output Açıklaması",
  "hint": "İpucu",
  "tags": ["Dizi"],
  "samples": [
    {{ "input": "...", "output": "...", "explanation": "..." }}
  ],
  "hidden_test_cases": [
    {{ "input": "...", "output": "..." }}
  ],
  "template": {{
    "C++": "// CODE",
    "Java": "// CODE",
    "Python3": "# CODE",
    "C": "// CODE",
    "JavaScript": "// CODE"
  }}
}}
{rag_context}
"""
        
        print("DEBUG: Initializing Groq client...")
        client = Groq(api_key=groq_api_key)
        
        print("DEBUG: Calling Groq API...")
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=6000,
            response_format={"type": "json_object"}
        )
        
        print("DEBUG: Groq API response received.")
        content = chat_completion.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"CRITICAL ERROR:\n{err_msg}")
        return {
            "error": True,
            "message": str(e),
            "traceback": err_msg
        }
