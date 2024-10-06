# https://www.youtube.com/watch?v=_ROckQHGHsU

import dspy
from dotenv import load_dotenv
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

assert load_dotenv("../../.env")

lm = dspy.LM(model="openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)


# Basic QA
predict = dspy.Predict("question -> answer")
prediction = predict(question="Who invented Telephone?")
print(prediction["answer"])


class QA(dspy.Signature):
    """Answer as succintly as possible"""

    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="Just the answer and nothing else.")


predict = dspy.Predict(QA)
prediction = predict(question="Who invented Telephone?")
print(prediction["answer"])


# Chain of thought
predict = dspy.Predict(QA)
prediction = predict(
    question="What is the birth places of the parents who's sons founded Google"
)
print(prediction["answer"])

predict = dspy.ChainOfThought(QA)
prediction = predict(
    question="What is the birth places of the parents who's sons founded Google"
)
print(prediction["answer"])
