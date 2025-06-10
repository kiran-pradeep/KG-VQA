from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from runner.task import QDVerdict, SubQuestion, Task
# from runner.database_manager import DatabaseManager
# from workflow.sql_meta_info import SQLMetaInfo

# import re, copy


class SystemState(BaseModel):
    """
    Represents the state of a graph during execution.

    Attributes:
        task (Task): The task associated with the graph state.
        tentative_schema (Dict[str, Any]): A dictionary representing the tentative schema.
        execution_history (List[Any]): A list representing the execution history.
    """

    executing_tool: str = ""
    task: Task
    sub_questions: Optional[List[SubQuestion]] = []
    execution_history: Optional[List[Any]] = []
    
    keywords: List[str] = []
    answer: Optional[str] = ""
    
    similar_columns: Optional[Dict[str, List[str]]] = {}
    
    # SQL_meta_infos: Dict[str, List[SQLMetaInfo]] = {}
    # SubSQL_meta_infos: List[Dict[str, List[SQLMetaInfo]]] = []
    # unit_tests: Dict[str, List[str]] = {}
    errors: Optional[Dict[str, str]] = {}
    qd_verdict: Optional[QDVerdict] = None
    support_indices: Optional[List[int]] = []

    def get_formatted_paragraphs(self) -> str:
        """
        Returns a formatted string of paragraphs from the task.

        Returns:
            str: A formatted string containing the paragraphs.
        """
        paragraphs = self.task.paragraphs
        formatted_paragraphs = ""
        for paragraph in paragraphs:
            formatted_paragraphs += f"Paragraph {paragraph.idx}: {paragraph.title}\n{paragraph.paragraph_text}\n\n"
        return formatted_paragraphs
    
    def get_formatted_sub_questions(self, sub_questions:List[SubQuestion]=[]) -> str:
        """
        Returns a formatted string of sub-questions.

        Returns:
            str: A formatted string containing the sub-questions.
        """
        if not sub_questions:
            print("!!!!! No sub-questions provided. Using ground-truth decompositions !!!!!")
            sub_questions = self.task.question_decomposition
        formatted_sub_questions = ""
        for idx, sub_question in enumerate(sub_questions):
            formatted_sub_questions += f"Sub-Question {idx + 1}: {sub_question.question}\n"
        return formatted_sub_questions
