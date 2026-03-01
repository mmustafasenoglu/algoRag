import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Optional RAG Context dependencies
try:
    from fastembed import TextEmbedding
    import re
    from pinecone import Pinecone
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

load_dotenv()

app = FastAPI(title="AlgoRAG API", description="AI Problem Generation Microservice")

class ProblemRequest(BaseModel):
    prompt: str

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
    
    if pinecone_api_key and RAG_AVAILABLE:
        try:
            # 1. Generate embedding for user prompt
            embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
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
                rag_context += "Use the following high-quality LeetCode examples to understand the expected format, tone, and complexity. Base the style and rigor of your generated problem on these examples.\n\n"
                for i, match in enumerate(search_results["matches"]):
                    meta = match.get("metadata", {})
                    title = meta.get("title", "Unknown")
                    diff = meta.get("difficulty", "Medium")
                    desc = meta.get("description", "")
                    
                    rag_context += f"Example {i+1}:\nTitle: {title}\nDifficulty: {diff}\nDescription:\n{desc}\n\n"
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"RAG Vector Search Error: {e}")
            
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
2. Use //TEMPLATE BEGIN ... //TEMPLATE END for the solution stub (DO NOT INCLUDE THE ANSWER).
3. Use //APPEND BEGIN ... //APPEND END for the main driver code that reads stdin and prints output.
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
        
        # --- Step 2: Validate edge test case outputs via a second Groq call ---
        hidden_cases = parsed_json.get("hidden_test_cases", [])
        edge_cases = hidden_cases[16:]  # items 17-20
        if edge_cases:
            validation_prompt = (
                "You are an expert computational engine. Below is a programming problem description, "
                "a solution template stub, and 4 large edge test case inputs designed for stress testing. "
                "Your task is to re-compute the EXACT CORRECT EXPECTED OUTPUT for these 4 inputs "
                "according to the problem rules. DO NOT write code; just provide the outputs in JSON.\n\n"
                f"Problem Title: {parsed_json.get('title')}\n"
                f"Description: {parsed_json.get('description')}\n\n"
                "Inputs to re-evaluate:\n"
            )
            for idx, ec in enumerate(edge_cases):
                validation_prompt += f"Input {idx+17}:\n{ec.get('input')}\n\n"
            
            validation_prompt += (
                "Respond ONLY with a valid JSON object matching this schema exactly:\n"
                "{\n"
                '  "corrected_outputs": [\n'
                '    {"output": "exact string output for Input 17"},\n'
                '    {"output": "exact string output for Input 18"},\n'
                '    {"output": "exact string output for Input 19"},\n'
                '    {"output": "exact string output for Input 20"}\n'
                "  ]\n"
                "}\n"
            )
            try:
                validation_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise algorithmic validator. You compute correct outputs for given inputs perfectly."
                        },
                        {
                            "role": "user",
                            "content": validation_prompt
                        }
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.1, # Extremely low temperature for deterministic validation
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )
                validation_content = validation_completion.choices[0].message.content
                validation_json = json.loads(validation_content)
                corrected_outputs = validation_json.get("corrected_outputs", [])
                
                # Apply corrections if lengths match
                if len(corrected_outputs) == len(edge_cases):
                    for idx in range(len(edge_cases)):
                        hidden_cases[16 + idx]["output"] = corrected_outputs[idx].get("output", edge_cases[idx].get("output"))
                parsed_json["hidden_test_cases"] = hidden_cases
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Edge case validation Groq API error: {e}")
                # If secondary validation fails, we proceed with the originally generated test cases instead of failing the whole request.
                pass 
                
        return parsed_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
