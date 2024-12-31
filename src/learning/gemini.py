from dotenv import load_dotenv
from litellm import completion

assert load_dotenv("../../.env")


response = completion(
    model="gemini/gemini-2.0-flash-exp",
    messages=[{"role": "user", "content": "How to unclog a drain?"}],
)

print(response.choices[0].message.content)
