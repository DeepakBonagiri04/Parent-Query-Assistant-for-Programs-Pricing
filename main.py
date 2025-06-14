import os
import requests
import json
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load API key
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found. Please set it in a .env file.")

# Load documents
loader = TextLoader("data/programs.md", encoding="utf-8")
documents = loader.load()

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create or load vectorstore
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if os.path.exists("chroma_store/index"):
    vectordb = Chroma(persist_directory="chroma_store", embedding_function=embedding)
else:
    vectordb = Chroma.from_documents(docs, embedding, persist_directory="chroma_store")

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Function for answering queries
def answer_query(user_query):
    relevant_docs = retriever.invoke(user_query)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/mixtral-8x7b-instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Only answer using the provided context. If it's not there, say you don't know."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {user_query}"
            }
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]

        # Save to feedback.json
        feedback = {
            "question": user_query,
            "context": context,
            "answer": answer
        }
        with open("feedback.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback, indent=2) + ",\n")

        return answer
    else:
        return f"❌ Error {response.status_code}: {response.text}"

# Gradio UI
gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(label="Ask your question"),
    outputs=gr.Textbox(label="Answer"),
    title="Parent Query Assistant",
    description="Ask about programs, pricing, or packages offered."
).launch()
