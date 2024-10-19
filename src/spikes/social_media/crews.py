from agents import ChatAgents
from crewai import Crew  # type: ignore
from tasks import ChatTasks


class ChatCrew:
    def __init__(self, topic: str, party: str, site: str):
        self.topic = topic
        self.party = party
        self.site = site

        self.query_agent = ChatAgents().search_query_generator_agent(party=party)
        self.chat_response_agent = ChatAgents().chat_response_agent(party=party)

    def get_response(self, context: str) -> str:
        query_task = ChatTasks().generate_search_query_task(
            topic=self.topic, context=context, party=self.party, agent=self.query_agent
        )

        chat_response_task = ChatTasks().generate_chat_response_task(
            topic=self.topic,
            site=self.site,
            party=self.party,
            context=context,
            agent=self.chat_response_agent,
        )

        crew = Crew(
            agents=[self.query_agent, self.chat_response_agent],
            tasks=[query_task, chat_response_task],
            verbose=True,
            memory=True,
        )
        response = crew.kickoff()
        return response.raw
