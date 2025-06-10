import logging, os
import traceback
from typing import Dict, List, Any

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from runner.task import QDVerdict
from workflow.system_state import SystemState
from workflow.agents.tool import Tool

class AnswerGenerator(Tool):
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
        self.answer = None
        self.support_indices = None
        

    def _run(self, state: SystemState):
        """
        Executes the question decomposition process.
        
        Args:
            state (SystemState): The current system state.

        Returns:
            None
        """
        sub_questions = state.sub_questions
        request_kwargs = {
            "CONTEXT": state.get_formatted_paragraphs(),
            "QUESTION": state.task.question,
            "SUB_QUESTIONS": state.get_formatted_sub_questions(sub_questions),
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
        self.answer = response[0]["final_answer"]
        self.supporting_paragraphs = response[0]["supporting_paragraphs"]
        self.sub_question_answers = response[0]["sub_question_answers"]


        self.support_indices = list(self.supporting_paragraphs.values())

        state.answer = self.answer
        state.support_indices = self.support_indices

        

    def _get_updates(self, state: SystemState) -> Dict:
        updates = {}
        updates["final_result"] = {
            "question": state.task.question,
            "chain_of_thought_reasoning": self.chain_of_thought_reasoning,
            "predicted_answer": self.answer,
            "actual_answer": state.task.answer,
            "answer_aliases": state.task.answer_aliases,
            "supporting_paragraphs": self.supporting_paragraphs,
            "sub_question_answers": self.sub_question_answers,
            "actual_support_indices": [
                                        paragraph.model_dump()["idx"]
                                        for paragraph in state.task.paragraphs
                                        if paragraph.model_dump()["is_supporting"]
                                    ],
        }
        return updates