import os

from chat_utils import write_to_file
from crews import ChatCrew
from dotenv import load_dotenv

assert load_dotenv("../../../.env")
os.environ["OPENAI_MODEL_NAME"] = ""

topic = "abortion"
TURN_COUNT = 3
context = []
filename = "outputs/context.txt"

liberal_debater = ChatCrew(
    topic=topic,
    party="liberal",
    site="msnbc.com",
)
conservative_debater = ChatCrew(
    topic=topic,
    party="conservative",
    site="foxnews.com",
)

for i in range(TURN_COUNT):
    response = liberal_debater.get_response(
        context="\n".join(context),
    )
    context.append(f"Liberal: {response}")
    response = conservative_debater.get_response(
        context="\n".join(context),
    )
    context.append(f"Conservative: {response}")

for c in context:
    print(c)

write_to_file(context, filename)
