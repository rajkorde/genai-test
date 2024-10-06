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
