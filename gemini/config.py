from google import genai
from google.genai import types

import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def generate(data, text):
    prompt = f"""
You are a professional forgery detection expert.

Task:
- Compare the provided reference data with the extracted text from a certificate.
- Determine the likelihood that the certificate is forged.
- If any critical details do not match, the forgery likelihood should be high.
- Respond with ONLY valid JSON in the exact format given.
- Do not add Markdown, code fences, or extra text.

Data: {data}
Extracted Text: {text}

Required JSON format:
{{
  "chances_of_forgery": "<number between 0 and 100>%",
  "accuracy": "<number between 0 and 100 of how much sure you are>%",
  "reason": "<concise reason why it may be fake>"
}}
"""

    client = genai.Client(api_key=GEMINI_API_KEY)
    model = "gemini-2.5-flash"

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        tools=[types.Tool(googleSearch=types.GoogleSearch())],
    )

    result = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        result += chunk.text

    # Remove Markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", result).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"Gemini returned invalid JSON after cleaning: {cleaned}")