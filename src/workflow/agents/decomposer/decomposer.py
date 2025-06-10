import traceback
from workflow.agents.agent import Agent
from workflow.agents.decomposer.tool_kit.question_decomposer import QuestionDecomposer
from workflow.agents.generator.tool_kit.answer_generator import AnswerGenerator



class Decomposer(Agent):
    """
    Agent responsible for selecting appropriate schemas based on the context.
    """
    
    def __init__(self, config: dict):
        """Initialize the tools needed for schema selection"""
        super().__init__(
            name="decomposer",
            task=("retrieve the most important entities and context relevant to the keywords of the question, through ",
                         "extracting keywords, retrieving entities, and retrieving context"),
            config=config,
        )

        self.tools = {}

        if "question_decomposer" in config["tools"]:
            print("Inside Question Decomposer!!!")
            self.tools["question_decomposer"] = QuestionDecomposer(**config["tools"]["question_decomposer"])
        else:
            print("config:\n", config)

        print("Inside Decomposer!!!")
        print(self.tools)
