import logging, os
import traceback
from typing import Dict, List, Any

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from runner.task import QDVerdict
from workflow.system_state import SystemState
from workflow.agents.tool import Tool

class QDEvaluator(Tool):
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
        self.relevance = None
        self.completeness = None
        self.granularity = None
        self.correctness = None
        self.coherence = None
        

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
        if "evaluation" in response[0]:
            self.correct = response[0]["evaluation"]
            print(f"Evaluation: {self.correct}")
            if self.correct.lower() == "correct":
                cumm_correctness = "correct"
                self.correctness = "correct"
                cumm_relevance = "relevant"
                self.relevance = "relevant"
                self.completeness = "complete"
                self.granularity = "optimal"
                cumm_coherence = "coherent"
                self.coherence = "coherent"
            else:
                cumm_correctness = "None"
                cumm_relevance = "None"
                self.completeness = "None"
                self.granularity = "None"
                cumm_coherence = "None"
        else:
            self.relevance = response[0]["relevance"]
            self.completeness = response[0]["completeness"]
            self.granularity = response[0]["granularity"]
            self.correctness = response[0]["correctness"]
            self.coherence = response[0]["coherence"]

            cumm_relevance = "relevant"
            for key, value in self.relevance.items():
                if value.lower() != "relevant":
                    cumm_relevance = "irrelevant"
                    break

            cumm_correctness = "correct"
            for key, value in self.correctness.items():
                if value.lower() != "correct":
                    cumm_correctness = "incorrect"
                    break

            cumm_coherence = "coherent"
            for key, value in self.coherence.items():
                if value.lower() != "coherent":
                    cumm_coherence = "incoherent"
                    break

        state.qd_verdict = QDVerdict(
            relevance=cumm_relevance,
            completeness=self.completeness,
            granularity=self.granularity,
            correctness=cumm_correctness,
            coherence=cumm_coherence,
            COT=self.chain_of_thought_reasoning,
        )
            

    def _get_updates(self, state: SystemState) -> Dict:
        updates = {
            "question": state.task.question,
            "ground_truth_sub_questions": [sub_question.question for sub_question in state.task.question_decomposition],
            "chain_of_thought_reasoning": self.chain_of_thought_reasoning,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "granularity": self.granularity,
            "correctness": self.correctness,
            "coherence": self.coherence,
        }
        return updates