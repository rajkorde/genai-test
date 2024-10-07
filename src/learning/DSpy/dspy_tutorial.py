# https://www.youtube.com/watch?v=_ROckQHGHsU

import dspy
from dotenv import load_dotenv
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot
from pydantic import BaseModel, Field

assert load_dotenv("../../.env")

# lm = dspy.OpenAI(model="openai/gpt-4o-mini")
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
class AnswerConfidence(BaseModel):
    answer: str = Field("Answer. 1-10 words.")
    confidence: float = Field("Your confidence score between 0-1.")


class QAWithConfidence(dspy.Signature):
    """Given user's question, answer it and also give a confidence value"""

    question = dspy.InputField()
    answer: AnswerConfidence = dspy.OutputField()


predict = dspy.TypedChainOfThought(QAWithConfidence)
prediction = predict(
    question="What is the birth places of the parents who's sons founded Google"
)
print(prediction["answer"])
lm.inspect_history(1)


class Answer(BaseModel):
    name: str = Field(description="Winner of the Man of the Match award in this year")
    year: int = Field(description="Year the finals was held. In format yyyy")


class QAList(dspy.Signature):
    """Given user's question, answer with JSON readable python list"""

    question = dspy.InputField()
    answer_list: list[Answer] = dspy.OutputField()


predict = dspy.TypedChainOfThought(QAList)
prediction = predict(
    question="Generate a list of 'Man of the match' winners of cricket T20 world cup finals from the time the tournament started"
)
print(prediction.answer_list)
lm.inspect_history(1)

# RAG
colbertv2_wiki17_abstracts = "http://20.102.90.50:2017/wiki17_abstracts"
dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

# from dspy.datasets import HotPotQA

# # Load the dataset.
# dataset = HotPotQA(
#     train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0
# )

# # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
# trainset = [x.with_inputs("question") for x in dataset.train]
# devset = [x.with_inputs("question") for x in dataset.dev]

# len(trainset), len(devset)


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


predict = RAG()
prediction = predict(question="Who won 2021 T20 World cup final?")
print(prediction.answer_list)
lm.inspect_history(1)


retrieve = dspy.Retrieve(k=3)
obj = retrieve(query_or_queries="Francis ford coppola")


# Few shot prompting
import ast
import random

import dspy
import pandas as pd

df = pd.read_csv("data/soccer_card_situations.csv")

trainset = df[:100]
testset = df[100:]

trainset = [
    dspy.Example(question=x.question, answer=x.answer).with_inputs("question")
    for x in trainset.itertuples()
]

testset = [
    dspy.Example(question=x.question, answer=x.answer).with_inputs("question")
    for x in testset.itertuples()
]


class RefreeAnswer(dspy.Signature):
    """Choose a card the referee shows, given a football situation"""

    question = dspy.InputField()
    answer = dspy.OutputField(description="Choose between: Yellow card, Red card")


class PredictionModel(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought(RefreeAnswer)

    def forward(self, question: str):
        return self.predict(question=question)


predict = PredictionModel()
prediction = predict(question="Spitting at the crowd")
print(prediction.answer)
lm.inspect_history(1)

from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match

evaluate_program = Evaluate(
    devset=testset,
    metric=answer_exact_match,
    num_threads=8,
    display_progress=True,
    display_table=True,
)
