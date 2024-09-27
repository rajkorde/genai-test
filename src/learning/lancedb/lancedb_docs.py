import lancedb
import pandas
import pyarrow as pa
from dotenv import load_dotenv

assert load_dotenv("../../.env")

uri = "data/sample-lancedb"
async_db = await lancedb.connect_async(uri)

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

db = lancedb.connect("temp/db")
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

db = lancedb.connect("test.db")
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
