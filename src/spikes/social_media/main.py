import os

from chat_utils import write_to_file
from crews import ChatCrew
from dotenv import load_dotenv

assert load_dotenv("../../../.env")

os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"


topic = "abortion"
TURN_COUNT = 5
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


def write_to_file(strings: list[str], file_path: str) -> None:
    try:
        with open(file_path, "w") as file:
            for string in strings:
                file.write(string + "\n")
    except OSError as e:
        print(f"An error occurred while writing to the file: {e}")


write_to_file(context, filename)
