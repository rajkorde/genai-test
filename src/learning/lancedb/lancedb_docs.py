import lancedb
import pandas
import pyarrow as pa

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
