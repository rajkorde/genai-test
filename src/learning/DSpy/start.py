import dspy
from dotenv import load_dotenv
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

assert load_dotenv("../../.env")


llm = dspy.OpenAI(model="gpt-4o-mini", max_tokens=2500)
dspy.settings.configure(lm=llm)

gsm8k = GSM8K()

gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]
print(gsm8k_trainset)


# define module
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("test -> answer")

    def forward(self, question: str):
        return self.prog(question=question)


# compile optimizer
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)

# Evaluate
evaluate = Evaluate(
    devset=gsm8k_devset,
    metric=gsm8k_metric,
    num_threads=4,
    display_progress=True,
    display_table=0,
)
evaluate(optimized_cot)
llm.inspect_history(n=8)


# DSPy RAG Tutorial
import dspy

colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)

# turbo = dspy.OpenAI(model="gpt-3.5-turbo")
# dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)

llm = dspy.OpenAI(model="gpt-4o-mini", max_tokens=2500)
dspy.settings.configure(lm=llm, rm=colbertv2_wiki17_abstracts)


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


# Ask any question you like to this simple RAG program.
my_question = "What castle did David Gregory inherit?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
predictor = RAG()
pred = predictor(question=my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")
