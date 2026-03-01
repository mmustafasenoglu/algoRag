import os
import json
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AlgoRAG API Debug")

# Optional RAG Global State
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding
        print("DEBUG: Loading embedding model...")
        _embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "debug_mode": True,
        "rag_ready": os.environ.get("ENABLE_RAG", "false").lower() == "true",
        "pinecone_key": bool(os.environ.get("PINECONE_API_KEY"))
    }

@app.post("/api/generate_problem")
async def generate_problem(request: Request):
    print("DEBUG: Root handler reached.")
    try:
        body = await request.json()
        prompt_text = body.get("prompt", "default")
        
        # Heavy imports inside to protect boot performance
        from groq import Groq
        
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            return {"error": True, "message": "GROQ_API_KEY missing"}

        # --- RAG Context Fetching (Optional) ---
        rag_context = ""
        ENABLE_RAG = os.environ.get("ENABLE_RAG", "false").lower() == "true"
        if ENABLE_RAG:
            try:
                print("DEBUG: Attempting RAG...")
                from pinecone import Pinecone
                pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                index = pc.Index("algoforge-rag")
                
                model = get_embedding_model()
                query_vector = list(model.embed([prompt_text]))[0].tolist()
                
                search_results = index.query(vector=query_vector, top_k=2, include_metadata=True)
                if search_results.get("matches"):
                    rag_context = "\n\n--- REFERENCE EXAMPLES ---\n"
                    for match in search_results["matches"]:
                        meta = match.get("metadata", {})
                        rag_context += f"Example: {meta.get('title')}\n{meta.get('description')}\n\n"
                    print("DEBUG: RAG context added.")
            except Exception as rag_err:
                print(f"DEBUG: RAG failed (non-fatal): {rag_err}")

        client = Groq(api_key=groq_api_key)
        
        system_prompt = f"""
Sana verilen konuya uygun, kurumsal kalitede bir algoritma problemi üret. 
Yanıtın TAMAMEN Türkçe olmalı. 
Yanıtın sadece aşağıdaki JSON formatında olmalı (Markdown blokları kullanma):

{{
  "title": "Problem Başlığı",
  "description": "HTML formatında detaylı problem hikayesi ve açıklaması",
  "input_description": "Girdi formatı açıklaması (HTML)",
  "output_description": "Çıktı formatı açıklaması (HTML)",
  "hint": "İpucu ve kısıtlar (HTML)",
  "tags": ["Dizi", "Dinamik Programlama"],
  "samples": [
    {{
      "input": "Örnek girdi",
      "output": "Örnek çıktı",
      "explanation": "Detaylı açıklama"
    }}
  ],
  "hidden_test_cases": [
    {{
      "input": "Gizli girdi",
      "output": "Gizli çıktı"
    }}
  ],
  "template": {{
    "C++": "//PREPEND BEGIN\\n#include <iostream>\\n//PREPEND END\\n\\n//TEMPLATE BEGIN\\nclass Solution {{\\npublic:\\n    int solve() {{ }}\\n}};\\n//TEMPLATE END\\n\\n//APPEND BEGIN\\nint main() {{ }}\\n//APPEND END",
    "Java": "// Java Sablonu...",
    "Python3": "# Python Sablonu...",
    "C": "// C Sablonu...",
    "JavaScript": "// JS Sablonu..."
  }}
}}

KURALLAR:
1. 'template' alanında mutlaka C++, Java, Python3, C ve JavaScript dilleri olmalı.
{rag_context}
"""
        
        print("DEBUG: Calling Groq...")
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=6000,
            response_format={"type": "json_object"}
        )
        
        parsed_json = json.loads(chat_completion.choices[0].message.content)
        
        # Validation
        required_fields = ["title", "description", "input_description", "output_description", "hint", "samples", "hidden_test_cases", "template"]
        for field in required_fields:
            if field not in parsed_json:
                parsed_json[field] = [] if "cases" in field or "samples" in field else {} if field == "template" else ""
                
        return parsed_json

    except Exception as e:
        err = traceback.format_exc()
        print(f"CRITICAL ERROR:\n{err}")
        return {
            "error": True, 
            "message": str(e),
            "traceback": err
        }
