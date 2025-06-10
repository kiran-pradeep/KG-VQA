import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Tuple

from utils.MuSiQue.evaluate import evaluate_single_instance
from utils.MuSiQue.metrics.answer import AnswerMetric
from utils.MuSiQue.metrics.group_answer_sufficiency import GroupAnswerSufficiencyMetric
from utils.MuSiQue.metrics.group_support_sufficiency import GroupSupportSufficiencyMetric
from utils.MuSiQue.metrics.support import SupportMetric
from workflow.system_state import SystemState


@dataclass
class Statistics:
    corrects: Dict[str, List[str]] = field(default_factory=dict)
    incorrects: Dict[str, List[str]] = field(default_factory=dict)
    errors: Dict[str, List[Union[Tuple[str, str], Tuple[str, str, str]]]] = field(default_factory=dict)
    total: Dict[str, int] = field(default_factory=dict)

    # QD Aspects
    incorrect_decomp: Dict[str, List[str]] = field(default_factory=dict) 
    irrelevant: Dict[str, List[str]] = field(default_factory=dict)
    incomplete: Dict[str, List[str]] = field(default_factory=dict)
    inadeq_decomp: Dict[str, List[str]] = field(default_factory=dict)
    incoherent: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self, all_metrics) -> Dict[str, Dict[str, Union[Dict[str, int], List[Tuple[str, str]]]]]:
        """
        Converts the statistics data to a dictionary format.

        Returns:
            Dict[str, Dict[str, Union[Dict[str, int], List[Tuple[str, str]]]]]: The statistics data as a dictionary.
        """
        return {
            "answer_metrics": {
                "answer_f1": round(all_metrics["answer_metric"].get_metric()[1], 3),
                "answer_em": round(all_metrics["answer_metric"].get_metric()[0], 3),
                "support_f1": round(all_metrics["support_metric"].get_metric()[1], 3),
            },
            "qd_counts": {
                key: {
                    "correct": len(self.corrects.get(key, [])),
                    "incorrect": len(self.incorrects.get(key, [])),
                    "error": len(self.errors.get(key, [])),
                    "total": self.total.get(key, 0)
                }
                for key in self.total
            },
            "QD_Aspects": {
                key: {
                    "incorrect_decomp": len(self.incorrect_decomp.get(key, [])),
                    "irrelevant": len(self.irrelevant.get(key, [])),
                    "incomplete": len(self.incomplete.get(key, [])),
                    "inadeq_decomp": len(self.inadeq_decomp.get(key, [])),
                    "incoherent": len(self.incoherent.get(key, []))
                }
                for key in self.total
            },
            "ids": {
                key: {
                    "correct": sorted(self.corrects.get(key, [])),
                    "incorrect": sorted(self.incorrects.get(key, [])),
                    "error": sorted(self.errors.get(key, [])),
                    "incorrect_decomp": sorted(self.incorrect_decomp.get(key, [])),
                    "irrelevant": sorted(self.irrelevant.get(key, [])),
                    "incomplete": sorted(self.incomplete.get(key, [])),
                    "inadeq_decomp": sorted(self.inadeq_decomp.get(key, [])),
                    "incoherent": sorted(self.incoherent.get(key, []))
                }
                for key in self.total
            }
        }

class StatisticsManager:
    def __init__(self, result_directory: str):
        """
        Initializes the StatisticsManager.

        Args:
            result_directory (str): The directory to store results.
        """
        self.result_directory = Path(result_directory)
        self.statistics = Statistics()


        self.all_metrics = {
            "answer_metric": AnswerMetric(),
            "support_metric": SupportMetric(),
            "group_answer_sufficiency_metric": GroupAnswerSufficiencyMetric(),
            "group_support_sufficiency_metric": GroupSupportSufficiencyMetric(),
        }

        # Ensure the statistics file exists
        self.statistics_file_path = self.result_directory / "-statistics.json"
        if not self.statistics_file_path.exists():
            self.statistics_file_path.touch()
            self.dump_statistics_to_file()
        else:
            data = json.load(open(self.statistics_file_path))
            # Extract the counts and populate total dictionary
            if "qd_counts" in data:
                for key, count_data in data["qd_counts"].items():
                    self.statistics.total[key] = count_data.get("total", 0)
            
            # Extract the ids data and populate the respective dictionaries
            if "ids" in data:
                for key, id_data in data["ids"].items():
                    # Convert correct ids back to tuples
                    if "correct" in id_data:
                        self.statistics.corrects[key] = [tuple(item) for item in id_data["correct"]]
                    
                    # Convert incorrect ids back to tuples
                    if "incorrect" in id_data:
                        self.statistics.incorrects[key] = [tuple(item) for item in id_data["incorrect"]]
                    
                    # Convert error ids back to tuples (which can be either 2 or 3 elements)
                    if "error" in id_data:
                        self.statistics.errors[key] = [tuple(item) for item in id_data["error"]]

                    # Convert incorrect_decomp ids back to tuples
                    if "incorrect_decomp" in id_data:
                        self.statistics.incorrect_decomp[key] = [tuple(item) for item in id_data["incorrect_decomp"]]
                    # Convert irrelevant ids back to tuples
                    if "irrelevant" in id_data:
                        self.statistics.irrelevant[key] = [tuple(item) for item in id_data["irrelevant"]]
                    # Convert incomplete ids back to tuples
                    if "incomplete" in id_data:
                        self.statistics.incomplete[key] = [tuple(item) for item in id_data["incomplete"]]
                    # Convert iadequate decomposition ids back to tuples
                    if "inadeq_decomp" in id_data:
                        self.statistics.inadeq_decomp[key] = [tuple(item) for item in id_data["inadeq_decomp"]]
                    # Convert incoherent ids back to tuples
                    if "incoherent" in id_data:
                        self.statistics.incoherent[key] = [tuple(item) for item in id_data["incoherent"]]

    def update_stats(self, question_id: str, validation_for: str, result: Dict[str, Any], state: SystemState=None):
        """
        Updates the statistics based on the validation result.

        Args:
            question_id (str): The question ID.
            validation_for (str): The validation context.
            result (Dict[str, Any]): The validation result.
        """
        if validation_for == "final_result":
            if not state:
                raise NotImplementedError
            predicted_dict = {
                "id": state.task.id,
                "predicted_answer": state.answer,
                "predicted_support_idxs": state.support_indices,
                "predicted_answerable": True,
            }
            actual_dict = state.task.model_dump()
            evaluate_single_instance(predicted_dict, actual_dict, self.all_metrics)

        else:
            exec_res = result["exec_res"]
            exec_err = result["exec_err"]

            self.statistics.total[validation_for] = self.statistics.total.get(validation_for, 0) + 1

            if exec_res == 1:
                if validation_for not in self.statistics.corrects:
                    self.statistics.corrects[validation_for] = []
                self.statistics.corrects[validation_for].append(question_id)
            else:
                if "!=" in exec_err:
                    if validation_for not in self.statistics.incorrects:
                        self.statistics.incorrects[validation_for] = []
                        self.statistics.incorrect_decomp[validation_for] = []
                        self.statistics.irrelevant[validation_for] = []
                        self.statistics.incomplete[validation_for] = []
                        self.statistics.inadeq_decomp[validation_for] = []
                        self.statistics.incoherent[validation_for] = []
                    self.statistics.incorrects[validation_for].append(question_id)
                    if "incorrect" in exec_err.lower() or "None" in exec_err.lower():
                        self.statistics.incorrect_decomp[validation_for].append(question_id)
                    if "irrelevant" in exec_err.lower() or "None" in exec_err.lower():
                        self.statistics.irrelevant[validation_for].append(question_id)
                    if "incomplete" in exec_err.lower() or "None" in exec_err.lower():
                        self.statistics.incomplete[validation_for].append(question_id)
                    if "under-decomposed" in exec_err.lower() or "over-decomposed" in exec_err.lower() or "None" in exec_err.lower():
                        self.statistics.inadeq_decomp[validation_for].append(question_id)
                    if "incoherent" in exec_err.lower() or "None" in exec_err.lower():
                        self.statistics.incoherent[validation_for].append(question_id)

                    
                else:
                    if validation_for not in self.statistics.errors:
                        self.statistics.errors[validation_for] = []
                    self.statistics.errors[validation_for].append((question_id, exec_err))

    def dump_statistics_to_file(self):
        """
        Dumps the current statistics to a JSON file.
        """
        with self.statistics_file_path.open('w') as f:
            json.dump(self.statistics.to_dict(self.all_metrics), f, indent=4)
