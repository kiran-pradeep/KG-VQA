import traceback
from typing import Dict

from runner.logger import Logger
from workflow.system_state import SystemState
from workflow.agents.tool import Tool


class QDAccuracy(Tool):
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

            print("Evaluating Question Decomposition Accuracy...")
            predicted_dict = state.qd_verdict
            actual_dict = (
                state.task.qd_verdict.model_dump() if state.task.qd_verdict else None
            )

            exec_res = 0
            exec_err = ""
            if not predicted_dict:
                exec_err += f"No prediction;"
            else:
                predicted_dict = predicted_dict.model_dump()
                for key in actual_dict:
                    if key == "COT":
                        continue  # Skip chain of thought for this evaluation
                    try:
                        if predicted_dict[key] == actual_dict[key]:
                            exec_res += 1
                        else:
                            exec_res = 0
                            exec_err += (
                                f"{key}:{predicted_dict[key]}!={actual_dict[key]}; "
                            )
                    except KeyError:
                        print(
                            f"KeyError: {key} not found in predicted_dict or actual_dict"
                        )

            self.evaluation_results["qd_result"] = {
                "exec_res": 0 if exec_res < 5 else 1,
                "exec_err": exec_err,
                "Question": state.task.question,
                "sub_questions": (
                    state.sub_questions
                    if state.sub_questions
                    else state.task.question_decomposition
                ),
                # "actual_label": actual_dict,
                "predicted_label": predicted_dict,
                "chain_of_thought": state.qd_verdict.COT if state.qd_verdict else None,
            }
        except:
            print(traceback.format_exc())

    def _get_updates(self, state: SystemState) -> Dict:
        return self.evaluation_results
