from typing import Annotated, List, Optional, Any, Dict, Union
from pydantic import BaseModel


class Paragraph(BaseModel):
    """
    Represents a paragraph with its content and metadata.

    Args:
        idx (int): The index of the paragraph.
        title (str): The title of the paragraph.
        paragraph_text (str): The text content of the paragraph.
        is_supporting (Optional[bool]): Indicates if the paragraph is supporting evidence, default is False.
    """

    idx: int
    title: str
    paragraph_text: str
    is_supporting: Optional[bool] = False


class SubQuestion(BaseModel):
    """
    Represents a sub-question with its content and metadata.

    Args:
        id (int): The unique identifier for the sub-question.
        question (str): The text of the sub-question.
        answer (Optional[str]): The answer to the sub-question, if available.
        paragraph_support_idx (Optional[int]): The index of the supporting paragraph, if applicable.
    """

    id: Optional[int] = -1
    question: str
    answer: Optional[str] = ""
    paragraph_support_idx: Optional[int] = 0


class QDVerdict(BaseModel):
    """
    Represents the verdict of a question decomposition.
    Attributes:
        correctness (str): The correctness of the question decomposition.
        relevance (str): The relevance of the question decomposition.
        completeness (str): The completeness of the question decomposition.
        granularity (str): The granularity of the question decomposition.
        coherence (str): The coherence of the question decomposition.
    """

    correctness: str = None
    relevance: str = None
    completeness: str = None
    granularity: str = None
    coherence: str = None
    COT: Optional[str] = None


class Task(BaseModel):
    """
    Represents a task with its associated metadata and content.

    Args:
        task_id (str): The unique identifier for the task.
        question_id (str): The unique identifier for the question associated with the task.
        paragraphs (List[Paragraph]): A list of paragraphs related to the task.
        question (str): The main question for the task.
        question_decomposition (Optional[List[SubQuestion]]): A list of sub-questions derived from the main question, if applicable.
        answer (Optional[str]): The answer to the main question, if available.
        answer_aliases (Optional[List[str]]): Alternative answers or aliases for the main answer, if any.
        answerable (Optional[bool]): Indicates if the question is answerable, default is None.
    """

    question_id: str = ""
    id: str
    paragraphs: List[Paragraph]
    question: str
    question_decomposition: Optional[List[SubQuestion]] = None
    answer: Optional[str] = None
    answer_aliases: Optional[List[str]] = None
    answerable: Optional[bool] = None
    qd_verdict: Optional[QDVerdict] = None
