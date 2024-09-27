from wikipediaapi import Wikipedia

wiki = Wikipedia("RAGBot/0.0", "en")

docs = [
    {"text": x, "category": "person"}
    for x in wiki.page("Hayao_Miyazaki").text.split("\n\n")
]

docs += [
    {"text": x, "category": "film"}
    for x in wiki.page("Spirited_Away").text.split("\n\n")
]

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

model_registry = get_registry().get("sentence-transformers")
model = model_registry.create(name="BAAI/bge-small-en-v1.5")


class Document(LanceModel):
    text: str = model.SourceField()
    vector: Vector(384) = model.VectorField()
    category: str


db = lancedb.connect("test.db")
tbl = db.create_table("my_table", schema=Document)

tbl.add(docs)

tbl.create_fts_index("text")

from lancedb.rerankers import CohereReranker

reranker = CohereReranker()

query = "What is Chihiro's new name given to her by the witch?"
