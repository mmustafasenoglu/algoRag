import os
import json
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AlgoRAG API Debug")

@app.get("/health")
async def health():
    return {"status": "ok", "debug_mode": True}

@app.post("/api/generate_problem")
async def generate_problem(request: Request):
    print("DEBUG: Root handler reached.")
    try:
        body = await request.json()
        prompt_text = body.get("prompt", "default")
        
        # Heavy imports inside to protect boot
        print("DEBUG: Importing dependencies...")
        from groq import Groq
        
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            return {"error": True, "message": "GROQ_API_KEY missing"}

        print("DEBUG: Calling Groq...")
        client = Groq(api_key=groq_api_key)
        
        system_prompt = "Bir algoritma problemi üret. Sadece JSON döndür."
        
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
        
        return json.loads(chat_completion.choices[0].message.content)

    except Exception as e:
        err = traceback.format_exc()
        print(f"CRITICAL ERROR:\n{err}")
        return {
            "error": True, 
            "message": str(e),
            "traceback": err
        }
