import traceback
from typing import Dict

from runner.logger import Logger
from workflow.system_state import SystemState
from workflow.agents.tool import Tool


class Accuracy(Tool):
    """
    Tool for evaluating the predicted SQL queries against the ground truth SQL query.
    """

    def __init__(self):
        super().__init__()

        self.evaluation_results = None

    def _run(self, state: SystemState):
        """
        Executes the evaluation process.

        Args:
            state (SystemState): The current system state.
        """
        try:
            self.evaluation_results = {}

            print("Evaluating Downstream Accuracy...")
            predicted_dict = {
                "id": state.task.id,
                "predicted_answer": state.answer,
                "predicted_support_idxs": state.support_indices,
                "predicted_answerable": True,
            }

            self.evaluation_results["final_result"] = {
                "Question": state.task.question,
                "sub_questions": (
                    state.sub_questions
                    if state.sub_questions
                    else state.task.question_decomposition
                ),
                "predicted": predicted_dict,
            }
        except:
            print(traceback.format_exc())

    def _get_updates(self, state: SystemState) -> Dict:
        return self.evaluation_results
