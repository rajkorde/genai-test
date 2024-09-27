import lancedb
import pandas
import pyarrow as pa
from dotenv import load_dotenv

assert load_dotenv("../../.env")
db_name = "test.db"

async_db = await lancedb.connect_async(db_name)

# Create table
data = [
    {"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
    {"vector": [5.9, 26.5], "item": "bar", "price": 20.0},
]

async_tbl = await async_db.create_table("my_table", data=data)

# add data
data = [
    {"vector": [1.3, 1.4], "item": "fizz", "price": 100.0},
    {"vector": [9.5, 56.2], "item": "buzz", "price": 200.0},
]
await async_tbl.add(data)

# KNN

await async_tbl.vector_search([100, 100]).limit(2).to_pandas()

# Create index - needs at least 256 vectors
await async_tbl.create_index("vector")

# Using embedding api
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

db = lancedb.connect(db_name)
func = get_registry().get("openai").create(name="text-embedding-ada-002")


class Words(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()


table = db.create_table("words", schema=Words, mode="overwrite")
table.add([{"text": "hello world"}, {"text": "goodbye world"}])

query = "greetings"
actual = table.search(query).limit(1).to_pydantic(Words)[0]
print(actual.text)

# hybrid search
import os

import lancedb
import openai
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

db = lancedb.connect(db_name)
embeddings = get_registry().get("openai").create()


class Documents(LanceModel):
    vector: Vector(embeddings.ndims()) = embeddings.VectorField()
    text: str = embeddings.SourceField()


table = db.create_table("documents", schema=Documents, mode="overwrite")

data = [
    {"text": "rebel spaceships striking from a hidden base"},
    {"text": "have won their first victory against the evil Galactic Empire"},
    {"text": "during the battle rebel spies managed to steal secret plans"},
    {"text": "to the Empire's ultimate weapon the Death Star"},
]

table.add(data)
table.create_fts_index("text")

results = table.search("concealed", query_type="hybrid")


# Vanilla RAG
import lancedb
import pandas as pd
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

with open("../../../data/rag/lease.txt", "r") as file:
    text_data = file.read()

import nltk

nltk.download("punkt")
import re

from nltk.tokenize import sent_tokenize


def recursive_text_splitter(text, max_chunk_length=1000, overlap=100):
    """
    Helper function for chunking text recursively
    """
    # Initialize result
    result = []

    current_chunk_count = 0
    separator = ["\n", " "]
    _splits = re.split(f"({separator})", text)
    splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]

    for i in range(len(splits)):
        if current_chunk_count != 0:
            chunk = "".join(
                splits[
                    current_chunk_count - overlap : current_chunk_count
                    + max_chunk_length
                ]
            )
        else:
            chunk = "".join(splits[0:max_chunk_length])

        if len(chunk) > 0:
            result.append("".join(chunk))
        current_chunk_count += max_chunk_length

    return result


chunks = recursive_text_splitter(text_data, max_chunk_length=100, overlap=10)
print("Number of Chunks: ", len(chunks))

import torch
from transformers import AutoModel, AutoTokenizer

# Choose a pre-trained model (e.g., BERT, RoBERTa, etc.)
# Load the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def embedder(chunk):
    """
    Helper function to embed chunk of text
    """
    # Tokenize the input text
    tokens = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)

    # Get the model's output (including embeddings)
    with torch.no_grad():
        model_output = model(**tokens)

    # Extract the embeddings
    embeddings = model_output.last_hidden_state[:, 0, :]
    embed = embeddings[0].numpy()
    return embed


embeds = []
for chunk in chunks:
    embed = embedder(chunks)
    embeds.append(embed)

import lancedb


def prepare_data(chunks, embeddings):
    """
    Helper function to prepare data to insert in LanceDB
    """
    data = []
    for chunk, embed in zip(chunks, embeddings):
        temp = {}
        temp["text"] = chunk
        temp["vector"] = embed
        data.append(temp)
    return data


def lanceDBConnection(chunks, embeddings):
    """
    LanceDB insertion
    """
    db = lancedb.connect(db_name)
    data = prepare_data(chunks, embeddings)
    table = db.create_table(
        "scratch",
        data=data,
        mode="overwrite",
    )
    return table


table = lanceDBConnection(chunks, embeds)

# Retriever
k = 5
question = "What is issue date of lease?"

# Embed Question
query_embedding = embedder(question)
# Semantic Search
result = table.search(query_embedding).limit(5).to_list()

context = [r["text"] for r in result]

base_prompt = """You are an AI assistant. Your task is to understand the user question, and provide an answer using the provided contexts. Every answer you generate should have citations in this pattern  "Answer [position].", for example: "Earth is round [1][2].," if it's relevant.

Your answers are correct, high-quality, and written by an domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer."

User question: {}

Contexts:
{}
"""

import openai

# llm
prompt = f"{base_prompt.format(question, context)}"
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,
    messages=[
        {"role": "system", "content": prompt},
    ],
)

print(response.choices[0].message.content)
