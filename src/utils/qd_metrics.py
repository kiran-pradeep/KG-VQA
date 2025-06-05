import argparse
import json
import matplotlib.pyplot as plt
import multiprocessing as mp
import nltk
import numpy as np
import os
import random
import re
import spacy
import string
import torch

from collections import Counter
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import List


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling numpy data types.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Global vars initialized inside worker processes
nlp = None
model = None
smoothie = SmoothingFunction().method4


def init_worker(roberta: bool = False):
    """
    Initialize the worker process with necessary resources.
    Args:
        roberta (bool): If True, use the SentenceTransformer model; otherwise, use spaCy.
    """
    global nlp, model, STOPWORDS
    if not roberta:
        # Load the spaCy model
        nlp = spacy.load("en_core_web_sm")
        nlp = spacy.load("en_core_web_md")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    else:
        # Load the sentence transformer model
        model = SentenceTransformer('distilroberta-base-msmarco-v2')  # or another preferred model
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    STOPWORDS = set(stopwords.words('english'))


def clean_sentence(text: str) -> str:
    """
    Cleans the input text by removing stopwords, punctuation, and tokens starting with '#'.
    Args:
        text (str): The input text to clean.
    Returns:
        str: The cleaned text.
    """
    # tokens = nltk.word_tokenize(text)
    start_with_hash_tokens = start_with_hash(text)
    tokens = text.split()
    tokens = [t for t in tokens if t.lower() not in start_with_hash_tokens]
    text = " ".join(tokens)
    text = remove_punctuation(text)
    
    return text


def start_with_hash(text: str) -> bool:
    """
    Checks if the text contains tokens that start with '#' followed by an integer or decimal number.
    Args:
        text (str): The input text to check.
    Returns:
        List[str]: A list of tokens that start with '#' followed by an integer or decimal number.
    """
    pattern = r"#\d+(\.\d+)?"
    full_matches = [match.group() for match in re.finditer(pattern, text)]

    return full_matches


def remove_punctuation(text: str) -> str:
    """
    Removes punctuation from the input text.
    Args:
        text (str): The input text from which to remove punctuation.
    Returns:
        str: The text with punctuation removed.
    """
    # Create a translation table that maps punctuation to None
    translator = str.maketrans('', '', string.punctuation)
    # Translate the text using the translation table
    return text.translate(translator)


def compute_bleu(candidate: str, references: List[str]) -> float:
    """
    Computes BLEU-1 score between a candidate sentence and a list of reference sentences.
    Args:
        candidate (str): The candidate sentence.
        references (List[str]): A list of reference sentences.
    Returns:
        float: The BLEU-1 score.
    """
    candidate = clean_sentence(candidate)
    references = [clean_sentence(ref) for ref in references]
    candidate_tokens = nltk.word_tokenize(candidate)
    references_tokens = [nltk.word_tokenize(ref) for ref in references]
    weights = (1.0, 0.0, 0.0, 0.0)  # Unigram BLEU
    return sentence_bleu(
                            references_tokens, 
                            candidate_tokens, 
                            smoothing_function=smoothie, 
                            weights=weights
                        )


def compute_rouge_1(candidate: str, reference: str) -> float:
    """
    Computes ROUGE-1 F1 score between a candidate and reference sentence.
    Args:
        candidate (str): The candidate sentence.
        reference (str): The reference sentence.
    Returns:
        float: The ROUGE-1 F1 score.
    """
    candidate = clean_sentence(candidate)
    reference = clean_sentence(reference)
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    score = scorer.score(reference, candidate)
    return score['rouge1'].fmeasure


def compute_precision(reference: str, candidate: str) -> float:
    """
    Computes precision (unigram-level) between a candidate and reference.
    Args:
        reference (str): The reference sentence.
        candidate (str): The candidate sentence.
    Returns:
        float: The precision score.
    """
    reference = clean_sentence(reference)
    candidate = clean_sentence(candidate)
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    
    # Clipped count: min(count in candidate, count in reference)
    clipped = sum(min(cand_counts[w], ref_counts[w]) for w in cand_counts)
    
    total_cand_unigrams = sum(cand_counts.values())
    
    if total_cand_unigrams == 0:
        return 0.0

    precision = clipped / total_cand_unigrams
    return precision


def compute_recall(reference: str, candidate: str) -> float:
    """
    Computes recall (unigram-level) between a candidate and reference.
    Args:
        reference (str): The reference sentence.
        candidate (str): The candidate sentence.
    Returns:
        float: The recall score.
    """
    reference = clean_sentence(reference)
    candidate = clean_sentence(candidate)
    ref_tokens = reference.split()
    cand_tokens = candidate.split()

    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)

    overlap = sum(min(ref_counts[w], cand_counts[w]) for w in ref_counts)

    total_ref_unigrams = sum(ref_counts.values())

    if total_ref_unigrams == 0:
        return 0.0

    recall = overlap / total_ref_unigrams
    return recall


def sentence_similarity(s1: str, s2: str) -> float:
    """
    Computes the semantic similarity between two sentences using either spaCy or SentenceTransformer.
    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.
    Returns:
        float: Similarity score between 0 and 1.
    """
    sim = None
    s1 = clean_sentence(s1)
    s2 = clean_sentence(s2)
    if nlp:
        # Get embeddings
        vec1 = nlp(s1).vector.reshape(1, -1)
        vec2 = nlp(s2).vector.reshape(1, -1)
        # Compute cosine similarity
        sim = cosine_similarity(vec1, vec2)[0][0]
    elif model:
        # Get embeddings
        emb1 = model.encode(s1, convert_to_tensor=True)
        emb2 = model.encode(s2, convert_to_tensor=True)
        # Compute cosine similarity
        sim = util.cos_sim(emb1, emb2).item()
    # return np.linalg.norm(vec1 - vec2)
    # return 1.0 - sim
    return sim


def calculate_scores(instance):
    """
    Calculate various scores for a given instance.
    Args:
        instance (dict): A dictionary containing the question, context, and sub-questions.
    Returns:
        dict: A dictionary containing the calculated scores.
    """
    question = instance["question"]
    context = instance["paragraphs"]
    sub_qs = instance["sub_questions"]
    concatenated_subqs = " ".join(sub_qs)
    # rouge1_score = compute_rouge_1(concatenated_subqs, question)
    recall = compute_recall(question, concatenated_subqs)
    question_precision = compute_precision(question, concatenated_subqs)
    context_precision = compute_precision(context, concatenated_subqs)
    question_distance = sentence_similarity(concatenated_subqs, question)
    context_distance = sentence_similarity(concatenated_subqs, context)

    subq_results = []

    for sub_q in sub_qs:
        # bleu_q = compute_bleu(sub_q, [question])
        # bleu_c = compute_bleu(sub_q, [context])
        precision_q = compute_precision(question, sub_q)
        precision_c = compute_precision(context, sub_q)
        recall_q = compute_recall(question, sub_q)

        dist_q = sentence_similarity(sub_q, question)
        dist_c = sentence_similarity(sub_q, context)

        subq_results.append({
            "sub_question": sub_q,
            # "question_bleu": bleu_q,
            # "context_bleu": bleu_c,
            # "rouge1": rouge_q,
            "question_precision": precision_q,
            "context_precision": precision_c,
            "recall": recall_q,
            "question_distance": dist_q,
            "context_distance": dist_c,
        })

    # avg_question_bleu = np.mean([x["question_bleu"] for x in subq_results])
    # avg_context_bleu = np.mean([x["context_bleu"] for x in subq_results])
    avg_question_distance = np.mean([x["question_distance"] for x in subq_results])
    avg_context_distance = np.mean([x["context_distance"] for x in subq_results])
    avg_question_precision = np.mean([x["question_precision"] for x in subq_results])
    avg_context_precision = np.mean([x["context_precision"] for x in subq_results])

    return {
        "id": instance["id"],
        "question": question,
        "sub_questions": subq_results,
        # "average_question_bleu": avg_question_bleu,
        # "average_context_bleu": avg_context_bleu,
        "average_question_similarity": avg_question_distance,
        "average_context_similarity": avg_context_distance,
        "question_similarity": question_distance,
        "context_similarity": context_distance,
        # "rouge1": rouge1_score,
        "average_question_precision": avg_question_precision,
        "average_context_precision": avg_context_precision,
        "question_precision": question_precision,
        "context_precision": context_precision,
        "recall": recall
    }

def preprocess_instance(raw_instance):
    """
    Preprocess a raw instance from the MuSiQue dataset.
    Args:
        raw_instance (dict): A dictionary containing the raw instance data.
    Returns:
        dict: A dictionary containing the preprocessed (formatted) instance data.
    """
    question = raw_instance.get("question", "")
    paragraphs = [f'{para["title"]}:\n{para["paragraph_text"]}\n\n' for para in raw_instance["paragraphs"]]
    formatted_paragraph = " ".join(paragraphs).strip()
    sub_questions = [subq['question'] for subq in raw_instance.get("question_decomposition", [])]
    return {
        "id": raw_instance.get("id", ""),
        "question": question,
        "paragraphs": formatted_paragraph,
        "sub_questions": sub_questions,
        "answer": raw_instance.get("answer", "")
    }

def perturb_data(instances, remove_relevant=0.0, add_irrelevant=0.0):
    """
    Perturb the dataset by removing relevant sub-questions or adding irrelevant ones.
    Args:
        instances (List[dict]): List of instances to perturb.
        remove_relevant (float): Proportion of relevant sub-questions to remove (0.0 - 1.0).
        add_irrelevant (float): Proportion of irrelevant sub-questions to add (0.0 - 1.0).
    Returns:
        List[dict]: List of perturbed instances.
    """
    if remove_relevant > 0.0:
        for inst in instances:
            n_to_remove = int(len(inst["sub_questions"]) * remove_relevant)
            inst["sub_questions"] = random.sample(inst["sub_questions"], len(inst["sub_questions"]) - n_to_remove)
        print(f"🔧 Applied removal of relevant sub-questions with proportion={remove_relevant}")
    if add_irrelevant > 0.0:
        # Gather all sub-questions from all instances
        all_subqs = [sq for inst in instances for sq in inst["sub_questions"]]
        for inst in instances:
            n_to_add = int(len(inst["sub_questions"]) * add_irrelevant)
            new_subqs = random.sample(all_subqs, n_to_add)
            inst["sub_questions"].extend(new_subqs)
        print(f"🔧 Applied addition of irrelevant sub-questions with proportion={add_irrelevant}")
    return instances


def make_output_dir(mode, perturbed=None):
    """
    Create an output directory for saving results.
    Args:
        mode (str): The mode of operation (e.g., 'dev', 'train').
        perturbed (str, optional): Perturbation type (e.g., 'remove_relevant=0.2').
    Returns:
        str: The path to the output directory.
    """
    if perturbed:
        mode = os.path.join(mode, perturbed)
    dir_name = os.path.join("results", "metrics", mode)
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def plot_graphs(metric_name: str, metric_values: List[float], output_dir: str):
    """
    Plot the distribution of a given metric and save the plot.
    Args:
        metric_name (str): The name of the metric to plot.
        metric_values (List[float]): The values of the metric to plot.
        output_dir (str): The directory where the plot will be saved.
    """
    plt.figure(figsize=(15, 10))
    plt.hist(metric_values, bins=100, color='red', edgecolor='black')
    plt.title(f"Distribution of {metric_name}")
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    plt.grid(True)

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    histogram_path = os.path.join(output_dir, "plots", f"{metric_name}_histogram.png")
    plt.savefig(histogram_path, bbox_inches='tight')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Mass-Force scorer for sub-questions")
    parser.add_argument("--mode", type=str, default="dev", help="Mode to run the scorer in (dev/train)")
    parser.add_argument("--roberta", type=bool, action="store_true", help="Use RoBERTa model for embeddings (default: False)")
    # group = parser.add_mutually_exclusive_group()
    parser.add_argument("--remove_relevant", type=float, default=0.0, help="Proportion of relevant sub-questions to remove (0.0 - 1.0)")
    parser.add_argument("--add_irrelevant", type=float, default=0.0, help="Proportion of irrelevant sub-questions to add (0.0 - 1.0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("🔧 Initializing...")
    jsonl_file = f"data/MuSiQue/updated_{args.mode}.jsonl"

    perturbed = None
    if args.remove_relevant > 0.0:
        perturbed = f"remove_relevant={args.remove_relevant}"
    elif args.add_irrelevant > 0.0:
        perturbed = f"add_irrelevant={args.add_irrelevant}"
    if args.remove_relevant > 0.0 and args.add_irrelevant > 0.0:
        raise ValueError("You can only specify one of --remove_relevant or --add_irrelevant, not both.")

    output_dir = make_output_dir(args.mode, args.w_bleu, args.w_sem, perturbed)
    with open(jsonl_file, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Preprocess
    processed_data = [preprocess_instance(x) for x in data]
    
    # Apply perturbation BEFORE multiprocessing
    if perturbed:
        print(f"🔧 Perturbing data with {perturbed}...")
        processed_data = perturb_data(
            processed_data,
            remove_relevant=args.remove_relevant,
            add_irrelevant=args.add_irrelevant
        )

    print("🧠 Running with multiprocessing...")
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=os.cpu_count(), initializer=init_worker, initargs=(args.roberta,)) as pool:
        results = list(tqdm(pool.imap(calculate_scores, processed_data), total=len(processed_data)))

    # Collect stats
    avg_metrics = [
                    "average_question_similarity", 
                    "average_context_similarity", 
                    "average_question_precision", 
                    "average_context_precision",
                    "recall", "question_similarity", 
                    "context_similarity", 
                    "question_precision", 
                    "context_precision"
                  ]
    
    print("📊 Plotting graphs...")
    statistics = {}
    for metric in avg_metrics:
        metric_values = [r[metric] for r in results]
        statistics[metric] = {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values)),
            "max": float(np.max(metric_values)),
            "min": float(np.min(metric_values))
        }
        plot_graphs(metric, metric_values, output_dir)

    statistics["num_questions"] = len(results)
    print("📂 Saving results...")

    with open(os.path.join(output_dir, "-scores.json"), "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    with open(os.path.join(output_dir, "-statistics.json"), "w") as f:
        json.dump(statistics, f, indent=4, cls=NumpyEncoder)

    print("✅ Completed:")
    print(f"- Sub-question results written to: {os.path.join(output_dir, '-scores.json')}")
    print(f"- Statistics written to: {os.path.join(output_dir, '-statistics.json')}")
