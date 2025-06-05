import traceback
from workflow.agents.agent import Agent

from workflow.agents.evaluator.tool_kit.qd_evaluator import QDEvaluator
from workflow.agents.evaluator.tool_kit.sql_decomposer import SQLDecomposer


class Evaluator(Agent):
    """
    Agent responsible for selecting appropriate schemas based on the context.
    """
    
    def __init__(self, config: dict):
        """Initialize the tools needed for schema selection"""
        super().__init__(
            name="evaluator",
            task="",
            config=config,
        )

        self.tools = {}

        if "qd_evaluator" in config["tools"]:
            print("Inside QD Evaluator!!!")
            self.tools["qd_evaluator"] = QDEvaluator(**config["tools"]["qd_evaluator"])

        print("Inside Evaluator!!!")

        
        # self.tools.update({
        #     "filter_column": FilterColumn(**config["tools"]["filter_column"]),
        #     "select_tables": SelectTables(**config["tools"]["select_tables"]),
        #     "select_columns": SelectColumns(**config["tools"]["select_columns"]),
        # })

        # if "schema_critic" in config["tools"]:
        #     self.tools["schema_critic"] = SchemaCritic(**config["tools"]["schema_critic"])
