import logging, os
import traceback
from typing import Dict, List, Any

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from runner.task import QDVerdict, SubQuestion
from workflow.system_state import SystemState
from workflow.agents.tool import Tool

class QuestionDecomposer(Tool):
    """
    Tool for selecting tables based on the specified mode and updating the tentative schema.
    """

    def __init__(self, template_name: str = None, engine_config: str = None, parser_name: str = None, sampling_count: int = 1):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name
        self.sampling_count = sampling_count

        self.chain_of_thought_reasoning = None
        self.sub_questions = None
        

    def _run(self, state: SystemState):
        """
        Executes the question decomposition process.
        
        Args:
            state (SystemState): The current system state.

        Returns:
            None
        """
        request_kwargs = {
            "CONTEXT": state.get_formatted_paragraphs(),
            "QUESTION": state.task.question,
        }
        
        response = async_llm_chain_call(
            prompt=get_prompt(template_name=self.template_name),
            engine=get_llm_chain(**self.engine_config),
            parser=get_parser(self.parser_name),
            request_list=[request_kwargs],
            step=self.tool_name,
            sampling_count=self.sampling_count,
        )[0]

        self.chain_of_thought_reasoning = response[0]["chain_of_thought_reasoning"]
        self.sub_questions = response[0]["sub_questions"]
        state.sub_questions = [SubQuestion(question=sub_question) for sub_question in self.sub_questions]

        

    def _get_updates(self, state: SystemState) -> Dict:
        updates = {
            "question": state.task.question,
            "chain_of_thought_reasoning": self.chain_of_thought_reasoning,
            "sub_questions": self.sub_questions,
        }
        return updates