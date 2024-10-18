import os

from agents import ChatAgents
from crewai import Crew  # type: ignore
from dotenv import load_dotenv
from tasks import ChatTasks

assert load_dotenv("../../../.env")

os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"


## Test space

topic = "abortion"
# context = [
#     "Abortion is morally wrong because it kills babies",
#     "Abortion provides a way for women to manage their own care",
# ]
context = []
party = "conservative"
site = "foxnews.com"

# party = "liberal"
# site = "msnbc.com"

query_agent = ChatAgents().search_query_generator_agent(party=party)
query_task = ChatTasks().generate_search_query_task(
    topic=topic, context="\n".join(context), party=party, agent=query_agent
)
chat_response_agent = ChatAgents().chat_response_agent(party=party)
chat_response_task = ChatTasks().generate_chat_response_task(
    topic=topic,
    site=site,
    party=party,
    context="\n".join(context),
    agent=chat_response_agent,
)


crew = Crew(
    agents=[query_agent, chat_response_agent],
    tasks=[query_task, chat_response_task],
    verbose=True,
    memory=True,
)
response = crew.kickoff()


# talkers


# Step 2. Executing a simple search query


# web_rag_tool = WebsiteSearchTool()


# topic = "abortion"
# sites = {
#     "liberal": ["msnbc.com", "nytimes.com"],
#     "neutral": ["twitter.com", "reddit.com"],
#     "conservative": ["foxnews.com"],
# }
# conservative_site = sites["conservative"][0]
# liberal_site = sites["liberal"][0]

# #
# crew = Crew(
#     agents=[researcher],
#     tasks=[
#         ChatTasks().generate_chat_response_task(
#             topic, conservative_site, "conservative", ""
#         )
#     ],
#     verbose=True,
#     memory=True,
# )

# response = crew.kickoff()
