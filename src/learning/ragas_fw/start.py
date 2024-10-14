import random

from datasets import load_dataset  # type: ignore
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate  # type: ignore
from ragas.metrics import (  # type: ignore
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator

assert load_dotenv("../../.env")

loader = DirectoryLoader("data/text", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()
for doc in docs:
    doc.metadata["filename"] = doc.metadata["source"]

# pick 100 random docs
NUM_DOCS = 100
random.seed(42)
docs = random.sample(docs, NUM_DOCS)

generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o-mini")  # ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

testset = generator.generate_with_langchain_docs(
    docs,
    test_size=5,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
    run_config=RunConfig(max_workers=2, max_retries=3),
)
