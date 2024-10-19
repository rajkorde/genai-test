from textwrap import dedent

from crewai import Agent, Task  # type: ignore


class ChatTasks:
    def generate_search_query_task(
        self, topic: str, context: str, party: str, agent: Agent
    ) -> Task:
        return Task(
            description=dedent(f"""
                Generate a web query string that would yield the best results to respond in a debate setting for a given topic. The query should be short (to be used for web search) and should help with the debate factoring in the topic and conversation so far. Assume you are representing a specific political party in the debate.

                Political Party: {party}
                Topic: {topic}
                Context: {context}

            """),
            expected_output="A short web search query",
            agent=agent,
        )

    def generate_chat_response_task(
        self, topic: str, site: str, party: str, context: str, agent: Agent
    ) -> Task:
        return Task(
            description=dedent(f"""
                Generate a reponse to a topic under debate. Assume you are representing the point of view of a specific political party. Before responding, read the points made so far in the conversation indicated in the context. If the context is empty, then assume this is the start of the debate. After reading the context, research the site listed below and provide a short chat response to a topic under discussion. 
                
                The tone should be conversational, not formal.
                Dont identity your political party. Just make a new counterpoint to discussion so far.
                The response should be 1-3 short sentences. 

                Topic: {topic}
                Site: {site}
                context: {context}
                Political party: {party}
            """),
            expected_output="A short chat response in text to a topic under discussion",
            agent=agent,
        )
