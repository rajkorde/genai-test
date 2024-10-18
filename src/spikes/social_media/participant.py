class Participant:
    def __init__(
        self, topic: str, name: str, party: str, site: str, context: list[str]
    ):
        self.name = name
        self.party = party
        self.site = site
        self.topic = topic
        self.context = context

    def run(self): ...
