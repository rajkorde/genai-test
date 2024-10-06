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
lm.inspect_history(1)


# Modules
class ChainOfThoughtCustom(dspy.Module):
    def __init__(self):
        self.cot1 = dspy.ChainOfThought("question -> step_by_step_thought")
        self.cot2 = dspy.ChainOfThought("question, thought -> answer")

    def forward(self, question: str):
        thought = self.cot1(question=question).step_by_step_thought
        answer = self.cot2(question=question, thought=thought).answer
        return dspy.Prediction(thought=thought, answer=answer)


predict = ChainOfThoughtCustom()
prediction = predict(
    question="What is the birth places of the parents who's sons founded Google"
)
print(prediction["answer"])
lm.inspect_history(1)

# Outputting typed predictors
