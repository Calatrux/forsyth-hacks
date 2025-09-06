# analyze_text.py
"""
This script takes a text file, extracts nodes and relationships using Gemini via LangChain, and outputs data for visualization with vis.js.
"""
from dotenv import load_dotenv
import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-pro", temperature=0)
graph_transformer = LLMGraphTransformer(llm=llm)

# Path to your input text file
text_path = 'input_text.txt'

with open(text_path, 'r') as f:
    text = f.read()

documents = [Document(page_content=text)]

# Use async for best performance (if in Jupyter, use asyncio.run or nest_asyncio)
import asyncio
async def extract_graph():
    return await graph_transformer.aconvert_to_graph_documents(documents)

graph_documents = asyncio.run(extract_graph())

# Output nodes and relationships for vis.js
import json
with open('graph_data.json', 'w') as f:
    json.dump([doc.dict() for doc in graph_documents], f, indent=2)

print("Graph data saved to graph_data.json. Use this file with vis.js for visualization.")
