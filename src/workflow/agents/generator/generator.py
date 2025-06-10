import traceback
from workflow.agents.agent import Agent
from workflow.agents.generator.tool_kit.answer_generator import AnswerGenerator



class Generator(Agent):
    """
    Agent responsible for selecting appropriate schemas based on the context.
    """
    
    def __init__(self, config: dict):
        """Initialize the tools needed for schema selection"""
        super().__init__(
            name="generator",
            task="",
            config=config,
        )

        self.tools = {}

        if "answer_generator" in config["tools"]:
            print("Inside Answer Generator!!!")
            self.tools["answer_generator"] = AnswerGenerator(**config["tools"]["answer_generator"])

        print("Inside Generator!!!")
