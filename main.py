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
You are an expert technical interviewer and competitive programming problem setter (like those at LeetCode or Codeforces). 
Your task is to generate a complete algorithm problem based on the user's prompt. 
You MUST output ONLY a pure, valid, raw JSON object (without Markdown block wrappers like ```json).

The JSON object MUST EXACTLY MATCH the following structure and schema:

{{
  "title": "A short, descriptive title",
  "description": "The detailed problem statement in English. Include context and constraints precisely.",
  "input_description": "Clear explanation of how the input is formatted. Example: 'The first line contains an integer T, the number of test cases. Valid formatting rule required.'",
  "output_description": "Clear explanation of what the output should be and its format.",
  "samples": [
    {{
      "input": "Sample raw input data exactly as it would hit stdin.",
      "output": "Sample raw output data exactly as it should be printed.",
      "explanation": "LeetCode style detailed explanation of how the answer is derived from the sample input."
    }}
  ],
  "hidden_test_cases": [
    {{
      "input": "Hidden raw input data 1",
      "output": "Hidden raw output data corresponding to the input 1"
    }}
  ],
  "tags": ["Tag1", "Tag2"],
  "template": {{
    "C++": "<FULL EXECUTABLE C++ TEMPLATE - see rules below>",
    "Java": "<FULL EXECUTABLE Java TEMPLATE - see rules below>",
    "Python3": "<FULL EXECUTABLE Python3 TEMPLATE - see rules below>",
    "C": "<FULL EXECUTABLE C TEMPLATE - see rules below>",
    "JavaScript": "<FULL EXECUTABLE JavaScript TEMPLATE - see rules below>"
  }}
}}

CRITICAL TEMPLATE RULES:
1. You MUST generate a "template" object containing EXACTLY FIVE keys: "C++", "Java", "Python3", "C", and "JavaScript".
2. The templates MUST follow a LeetCode-style structure with THREE sections:
   - //PREPEND BEGIN ... //PREPEND END : necessary imports and global declarations only
   - //TEMPLATE BEGIN ... //TEMPLATE END : the function/class signature the user will implement. This MUST contain ONLY the method signature and a placeholder body (// TODO or pass). ABSOLUTELY DO NOT write the actual algorithm or solution logic here — the user must implement it themselves.
   - //APPEND BEGIN ... //APPEND END : a complete, RUNNABLE driver that reads stdin, calls the user's function, and prints the result.
3. WARNING: The //TEMPLATE section must NEVER contain a working solution. If you put real algorithm logic inside //TEMPLATE, users will see the answer and the problem becomes useless. Only a stub/skeleton is allowed.
4. The //APPEND section is CRITICAL and MUST NOT be empty. Tailor it exactly to the problem's input/output format.
5. BEFORE writing the APPEND driver and test cases, decide on ONE fixed input format (e.g., "first line: n, second line: n space-separated integers") and use it CONSISTENTLY in both the APPEND stdin reading code AND every single test case input. A mismatch will cause Runtime Errors.
6. Escape ALL newline characters as `\\n` inside the JSON template strings.
7. ABSOLUTELY NO unescaped double quotes inside the JSON string values.

TEMPLATE EXAMPLE (for a graph shortest-path problem in C++):
"C++": "//PREPEND BEGIN\\n#include <iostream>\\n#include <vector>\\n#include <queue>\\n#include <limits>\\nusing namespace std;\\n//PREPEND END\\n\\n//TEMPLATE BEGIN\\nclass Solution {{\\npublic:\\n    vector<int> solve(int n, vector<vector<int>>& edges, int src) {{\\n        // TODO: Implement here\\n    }}\\n}};\\n//TEMPLATE END\\n\\n//APPEND BEGIN\\nint main() {{\\n    ios_base::sync_with_stdio(false);\\n    cin.tie(NULL);\\n    int n, m, s;\\n    cin >> n >> m >> s;\\n    vector<vector<int>> edges(m, vector<int>(3));\\n    for (int i = 0; i < m; i++) cin >> edges[i][0] >> edges[i][1] >> edges[i][2];\\n    Solution sol;\\n    vector<int> res = sol.solve(n, edges, s);\\n    for (int i = 0; i < n; i++) {{\\n        if (i) cout << ' ';\\n        cout << (res[i] == INT_MAX ? -1 : res[i]);\\n    }}\\n    cout << endl;\\n    return 0;\\n}}\\n//APPEND END"

CRITICAL TEST CASE RULES:
1. You MUST generate EXACTLY 20 items in the `hidden_test_cases` array.
2. The first 16 test cases should cover standard functionality and constraints, including: empty/single-element inputs, all-same elements, already sorted, reverse sorted, and various sizes.
3. The LAST 4 test cases (items 17 to 20) MUST be LARGE stress tests: generate inputs with at least 3000-5000 elements to ensure O(n^2) solutions exceed the time limit. These must be valid, real inputs with correctly computed outputs.
4. ABSOLUTELY NO UNESCAPED QUOTES inside JSON values. If you use quotes inside the text, use single quotes (') or escape double quotes (\\").
5. MOST CRITICAL: The `input` field of EVERY test case MUST be in the EXACT SAME FORMAT that the //APPEND driver reads from stdin. For example, if the driver reads `n` on the first line and then `n` numbers on the second line, every test case input MUST have exactly those two lines in that order. The test cases and the APPEND driver are executed together — any format mismatch will cause a Runtime Error on ALL test cases.
{rag_context}
"""
    client = Groq(api_key=groq_api_key)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
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
