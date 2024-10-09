from datasets import load_dataset  # type: ignore
from dotenv import load_dotenv
from ragas import evaluate  # type: ignore
from ragas.metrics import (  # type: ignore
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

assert load_dotenv("../../.env")

amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")


result = evaluate(
    amnesty_qa["eval"],
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)
