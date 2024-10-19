from textwrap import dedent

from crewai import LLM, Agent  # type: ignore
from crewai_tools import WebsiteSearchTool, tool  # type: ignore
from duckduckgo_search import DDGS


@tool("Get Top 3 Web Search Results")  # type: ignore
def web_search_top_results(topic: str, site: str) -> list[str]:
    """given a query and a root site, return the top three result links from duckduckgo"""
    topic = f"{topic} site:{site}"
    results = DDGS().text(topic, max_results=3)
    return [result["href"] for result in results] if results else []


class ChatAgents:
    def __init__(self, use_openai: bool = True):
        if use_openai:
            self.llm = LLM(model="gpt-4o-mini", temperature=0.7)
        else:
            self.llm = LLM(
                model="ollama/llama3.2",
                base_url="http://localhost:11434",
                temperature=0.7,
            )

    def search_query_generator_agent(self, party: str) -> Agent:
        return Agent(
            role=dedent(
                """A competent researcher who knows how to create web query string to get the best web results"""
            ),
            goal=dedent(
                """Create a web query string that would yield the best results to respond in a debate setting for a given topic. Assume you belong to a specific political party. The query is generated using the debate topic and the past few points made in the debate.
                You should not repeat what has already been discussed."""
            ),
            backstory=f"An opinionated political nerd from {party} party",
            verbose=True,
            llm=self.llm,
        )

    def chat_response_agent(self, party: str):
        return Agent(
            role=dedent(
                """A researcher who searches the web given a query and reads the links to create a response for the debate"""
            ),
            goal=dedent("""Provide a chat response to on a topic in 1-2 short sentences. Use the search query to get the top links. Read the links to create a chat reponse on a given topic. Assume you belong to a specific political party, factoring in the discussion so far.
            Dont repeat what has already been discussed.
            Respond in first person"""),
            backstory=f"An opinionated political nerd representing the {party} party.",
            tools=[web_search_top_results, WebsiteSearchTool()],
            verbose=True,
            llm=self.llm,
        )
