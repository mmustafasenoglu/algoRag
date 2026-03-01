import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Optional RAG Context dependencies
embedding_model = None
try:
    from fastembed import TextEmbedding
    import re
    from pinecone import Pinecone
    RAG_AVAILABLE = True
    # Load model globally to avoid OOM and slow startup on every request
    print("Loading embedding model...")
    embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"RAG dependencies or model loading failed: {e}")
    RAG_AVAILABLE = False

load_dotenv()

app = FastAPI(title="AlgoRAG API", description="AI Problem Generation Microservice")

class ProblemRequest(BaseModel):
    prompt: str

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "groq_configured": bool(os.environ.get("GROQ_API_KEY")),
        "pinecone_configured": bool(os.environ.get("PINECONE_API_KEY")),
        "rag_available": RAG_AVAILABLE
    }

@app.post("/api/generate_problem")
async def generate_problem(request: ProblemRequest):
    prompt_text = request.prompt
    
    if not prompt_text:
        raise HTTPException(status_code=400, detail="Missing required parameter: prompt")
        
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="AI service is not configured (Missing GROQ_API_KEY)")
        
    # Optional RAG Context
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    rag_context = ""
    
    if pinecone_api_key and RAG_AVAILABLE and embedding_model:
        try:
            # 1. Generate embedding for user prompt
            query_vector = list(embedding_model.embed([prompt_text]))[0].tolist()
            
            # 2. Search Pinecone vector database
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index("algoforge-rag")
            
            search_results = index.query(
                vector=query_vector,
                top_k=3,
                include_metadata=True
            )
            
            # 3. Format RAG context
            if search_results.get("matches"):
                rag_context = "\n\n--- REFERENCE EXAMPLES (For Inspiration & Quality Standards) ---\n"
                rag_context += "Use the following high-quality examples to understand the expected format.\n\n"
                for i, match in enumerate(search_results["matches"]):
                    meta = match.get("metadata", {})
                    title = meta.get("title", "Unknown")
                    diff = meta.get("difficulty", "Medium")
                    desc = meta.get("description", "")
                    rag_context += f"Example {i+1}:\nTitle: {title}\nDifficulty: {diff}\nDescription:\n{desc}\n\n"
        except Exception as e:
            print(f"RAG Error (Non-fatal): {e}")
            
    # Instruct AI to construct the complete API payload
    system_prompt = f"""
You are an expert competitive programming problem setter. 
Your task is to generate a complete algorithm problem based on the user's prompt. 
Your response MUST be in Turkish.
Your response MUST be a valid JSON object.

The JSON object MUST have the following strict structure:
{{
  "title": "A short, descriptive title",
  "description": "HTML formatted detailed story and description of the problem",
  "input_description": "HTML formatted explanation of the input",
  "output_description": "HTML formatted explanation of the expected output",
  "hint": "HTML formatted hints or constraints",
  "samples": [
    {{
      "input": "Sample raw input data",
      "output": "Sample raw output data",
      "explanation": "LeetCode style detailed explanation"
    }}
  ],
  "hidden_test_cases": [
    {{
      "input": "Hidden raw input 1",
      "output": "Hidden raw output 1"
    }}
  ],
  "tags": ["Tag1", "Tag2"],
  "template": {{
    "C++": "<FULL EXECUTABLE C++ TEMPLATE>",
    "Java": "<FULL EXECUTABLE Java TEMPLATE>",
    "Python3": "<FULL EXECUTABLE Python3 TEMPLATE>",
    "C": "<FULL EXECUTABLE C TEMPLATE>",
    "JavaScript": "<FULL EXECUTABLE JavaScript TEMPLATE>"
  }}
}}

CRITICAL TEMPLATE RULES (Apply to all 5 languages):
1. Use //PREPEND BEGIN ... //PREPEND END for imports.
2. Use //TEMPLATE BEGIN ... //TEMPLATE END for the solution stub.
3. Use //APPEND BEGIN ... //APPEND END for the main driver code.
4. Ensure the APPEND I/O logic exactly matches the test case input format.

CRITICAL TEST CASE RULES:
1. EXPLICITLY generate 20 items in `hidden_test_cases`.
2. Items 17-20 MUST be large stress tests (arrays of 3000+ elements).
{rag_context}
"""
    client = Groq(api_key=groq_api_key)
    try:
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
        
        generated_content = chat_completion.choices[0].message.content
        parsed_json = json.loads(generated_content)
        
        # Validate critical fields
        required = ["title", "description", "input_description", "output_description", "hint", "samples", "hidden_test_cases", "template"]
        for field in required:
            if field not in parsed_json:
                parsed_json[field] = [] if field in ["samples", "hidden_test_cases", "tags"] else {} if field == "template" else ""

        return parsed_json
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"API ERROR: {error_details}")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
