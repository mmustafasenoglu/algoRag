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
    print(f"Received request: {request.prompt[:50]}...")
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing")
        
    rag_context = ""
    if RAG_ENABLED and embedding_model:
        try:
            print("Fetching RAG context...")
            pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
            index = pc.Index("algoforge-rag")
            query_vector = list(embedding_model.embed([request.prompt]))[0].tolist()
            search_results = index.query(vector=query_vector, top_k=3, include_metadata=True)
            
            if search_results.get("matches"):
                rag_context = "\n\n--- REFERENCE EXAMPLES ---\n"
                for i, match in enumerate(search_results["matches"]):
                    meta = match.get("metadata", {})
                    rag_context += f"Example {i+1}:\nTitle: {meta.get('title')}\nDescription: {meta.get('description')}\n\n"
        except Exception as e:
            print(f"RAG Error: {e}")
            
    system_prompt = f"""
Sana verilen konuya uygun, kurumsal kalitede bir algoritma problemi üret. 
Yanıtın TAMAMEN Türkçe olmalı. 
Yanıtın sadece aşağıdaki JSON formatında olmalı:

{{
  "title": "Problem Başlığı",
  "description": "HTML formatında problem açıklaması",
  "input_description": "Input formatı açıklaması",
  "output_description": "Output formatı açıklaması",
  "hint": "İpuçları",
  "tags": ["Dizi", "Algoritma"],
  "samples": [
    {{ "input": "...", "output": "...", "explanation": "..." }}
  ],
  "hidden_test_cases": [
    {{ "input": "...", "output": "..." }}
  ],
  "template": {{
    "C++": "// CODE HERE",
    "Java": "// CODE HERE",
    "Python3": "# CODE HERE",
    "C": "// CODE HERE",
    "JavaScript": "// CODE HERE"
  }}
}}

KURALLAR:
1. Template'ler //PREPEND, //TEMPLATE, //APPEND bloklarını içermeli.
2. Toplam 20 tane 'hidden_test_cases' üret.
3. JSON dışında hiçbir metin yazma.
{rag_context}
"""
    
    client = Groq(api_key=groq_api_key)
    try:
        print("Calling Groq API...")
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
        
        print("Response received from Groq.")
        return json.loads(chat_completion.choices[0].message.content)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        # Return a JSON object even for errors so the frontend can display the message
        return {
            "error": True,
            "message": str(e)
        }
