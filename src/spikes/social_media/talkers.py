from textwrap import dedent

from crewai import Agent, Crew, Task  # type: ignore
from crewai_tools import WebsiteSearchTool, tool  # type: ignore
from dotenv import load_dotenv
from duckduckgo_search import DDGS

assert load_dotenv("../../../.env")


# Step 2. Executing a simple search query


@tool("Get Top 3 Web Search Results")  # type: ignore
def web_search_top_results(topic: str, site: str) -> list[str]:
    """given a query and a root site, return the top three result links from duckduckgo"""
    topic = f"{topic} site:{site}"
    results = DDGS().text(topic, max_results=3)
    return [result["href"] for result in results] if results else []


web_rag_tool = WebsiteSearchTool()

researcher = Agent(
    role="A researcher who uses web search to get links to relevant articles and reads those articles to create a chat reponse on a given topic. The chat response should be 1-3 sentences only.",
    goal="Provide a chat response to on a topic in 1-3 sentences",
    backstory="An opinianated political nerd",
    tools=[web_rag_tool, web_search_top_results],
    verbose=True,
)


class ChatTasks:
    def generate_chat_response_task(
        self, topic: str, site: str, party: str, context: str
    ) -> Task:
        return Task(
            description=dedent(f"""
                Generate a reponse to a topic under debate from the point of view of a specific political party. Before responding, you read the points made so far in the conversation indicated in the context. If the context is empty, then assume this is the start of the debate. After reading the context, research the sites listed below and provide a short chat response to a topic under discussion. The response should be 1-3 sentences long.

                Topic: {topic}
                Site: {site}
                context: {context}
                Political party: {party}
            """),
            expected_output="A short chat response in text to a topic under discussion",
            agent=researcher,
        )


topic = "abortion"
sites = {
    "liberal": ["msnbc.com", "nytimes.com"],
    "neutral": ["twitter.com", "reddit.com"],
    "conservative": ["foxnews.com"],
}
conservative_site = sites["conservative"][0]
liberal_site = sites["liberal"][0]

#
crew = Crew(
    agents=[researcher],
    tasks=[
        ChatTasks().generate_chat_response_task(
            topic, conservative_site, "conservative", ""
        )
    ],
    verbose=True,
    memory=True,
)

response = crew.kickoff()
