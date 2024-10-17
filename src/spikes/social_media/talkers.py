from dotenv import load_dotenv
from duckduckgo_search import DDGS

assert load_dotenv("../../../.env")


# Step 2. Executing a simple search query
topic = "abortion site:foxnews.com"

results = DDGS().text(topic, max_results=3)
