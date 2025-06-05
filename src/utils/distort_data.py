import argparse
import os
import json
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import spacy
import multiprocessing as mp


class NumpyEncoder(json.JSONEncoder):
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
smoothie = SmoothingFunction().method4

def init_worker():
    global nlp
    nlp = spacy.load("en_core_web_md")
    nltk.download('punkt', quiet=True)

def compute_bleu(candidate: str, references: List[str]) -> float:
    candidate_tokens = nltk.word_tokenize(candidate)
    references_tokens = [nltk.word_tokenize(ref) for ref in references]
    weights = (1.0, 0.0, 0.0, 0.0)  # Unigram BLEU
    return sentence_bleu(
                            references_tokens, 
                            candidate_tokens, 
                            smoothing_function=smoothie, 
                            weights=weights
                        )

def semantic_distance(s1: str, s2: str) -> float:
    vec1 = nlp(s1).vector.reshape(1, -1)
    vec2 = nlp(s2).vector.reshape(1, -1)
    sim = cosine_similarity(vec1, vec2)[0][0]
    return 1.0 - sim

def calculate_mass_force(args_tuple):
    instance, w_bleu, w_sem = args_tuple
    question = instance["question"]
    context = instance["paragraphs"]
    sub_qs = instance["sub_questions"]

    subq_results = []

    for sub_q in sub_qs:
        bleu_q = compute_bleu(sub_q, [question])
        bleu_c = compute_bleu(sub_q, [context])
        mass = w_bleu * bleu_q + (1 - w_bleu) * bleu_c

        dist_q = semantic_distance(sub_q, question)
        dist_c = semantic_distance(sub_q, context)
        distance = w_sem * dist_q + (1 - w_sem) * dist_c

        force = mass / (distance**2 + 1e-6)

        subq_results.append({
            "sub_question": sub_q,
            "mass": mass,
            "distance": distance,
            "question_bleu": bleu_q,
            "context_bleu": bleu_c,
            "question_distance": dist_q,
            "context_distance": dist_c,
            "force": force
        })

    avg_force = np.mean([x["force"] for x in subq_results])
    avg_question_bleu = np.mean([x["question_bleu"] for x in subq_results])
    avg_context_bleu = np.mean([x["context_bleu"] for x in subq_results])
    avg_question_distance = np.mean([x["question_distance"] for x in subq_results])
    avg_context_distance = np.mean([x["context_distance"] for x in subq_results])
    avg_mass = np.mean([x["mass"] for x in subq_results])
    avg_distance = np.mean([x["distance"] for x in subq_results])

    return {
        "id": instance["id"],
        "question": question,
        "sub_questions": subq_results,
        "average_force": avg_force,
        "average_mass": avg_mass,
        "average_distance": avg_distance,
        "average_question_bleu": avg_question_bleu,
        "average_context_bleu": avg_context_bleu,
        "average_question_distance": avg_question_distance,
        "average_context_distance": avg_context_distance
    }

import random

def generate_distractor(question: str, n: int = 1) -> List[str]:
    distractors = [
        "What is the capital of France?",
        "How many legs does a spider have?",
        "Who wrote 'To Kill a Mockingbird'?",
        "What is the boiling point of water?",
        "When did World War II end?",
        "What is the speed of light?",
        "What is 2 + 2?",
        "Name a programming language.",
        "Where is the Eiffel Tower located?",
        "Who painted the Mona Lisa?"
    ]
    return random.sample(distractors, k=min(n, len(distractors)))

def preprocess_instance(raw_instance, add_noise=0, remove_ratio=0.0):
    question = raw_instance.get("question", "")
    paragraphs = [f'{para["title"]}:\n{para["paragraph_text"]}\n\n' for para in raw_instance["paragraphs"]]
    formatted_paragraph = " ".join(paragraphs).strip()
    sub_questions = [subq['question'] for subq in raw_instance.get("question_decomposition", [])]

    # Simulate removal of sub-questions
    if remove_ratio > 0.0 and sub_questions:
        num_to_remove = int(len(sub_questions) * remove_ratio)
        try:
            sub_questions = random.sample(sub_questions, len(sub_questions) - num_to_remove)
        except ValueError:
            print(f"[WARNING] Not enough sub-questions to remove {num_to_remove}. Keeping all {len(sub_questions)} sub-questions.")
            raise ValueError("Not enough sub-questions to remove the specified ratio.")

    # Add distractor sub-questions
    if add_noise > 0:
        sub_questions += generate_distractor(question, add_noise)

    return {
        "id": raw_instance.get("id", ""),
        "question": question,
        "paragraphs": formatted_paragraph,
        "sub_questions": sub_questions,
        "answer": raw_instance.get("answer", "")
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Mass-Force scorer for sub-questions")
    parser.add_argument("--mode", type=str, default="dev", help="Mode to run the scorer in (dev/train)")
    parser.add_argument("--w_bleu", type=float, default=1.0, help="Weight for BLEU score (question vs context)")
    parser.add_argument("--w_sem", type=float, default=0.4, help="Weight for semantic distance (question vs context)")
    parser.add_argument("--add_noise", type=int, default=2, help="Number of random distractor sub-questions to add")
    parser.add_argument("--remove_ratio", type=float, default=0.5, help="Fraction of sub-questions to randomly remove")
    return parser.parse_args()


def make_output_dir(mode, w_bleu, w_sem):
    dir_name = f"results/{mode}_corrupted/w_bleu={w_bleu}_w_sem={w_sem}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def plot_graphs(metric_name: str, metric_values: List[float], output_dir: str):
    import matplotlib.pyplot as plt

    print(f"[DEBUG] type(metric_values)={type(metric_values)}, len={len(metric_values)}")
    print(f"[DEBUG] Sample values: {metric_values[:5]}")

    plt.figure(figsize=(15, 10))
    plt.hist(metric_values, bins=100, color='red', edgecolor='black')
    plt.title(f"Distribution of {metric_name}")
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    plt.grid(True)

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    histogram_path = os.path.join(output_dir, "plots", f"{metric_name}_histogram.png")
    plt.savefig(histogram_path)
    plt.close()




if __name__ == "__main__":
    args = parse_args()
    print("🔧 Initializing...")
    jsonl_file = f"data/MuSiQue/updated_{args.mode}.jsonl"

    output_dir = make_output_dir(args.mode, args.w_bleu, args.w_sem)
    with open(jsonl_file, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Preprocess
    # In main():
    processed_data = [
        preprocess_instance(x, add_noise=args.add_noise, remove_ratio=args.remove_ratio)
        for x in data
    ]

    print("🧠 Running with multiprocessing...")
    with mp.Pool(processes=os.cpu_count(), initializer=init_worker) as pool:
        input_data = [(instance, args.w_bleu, args.w_sem) for instance in processed_data]
        results = list(tqdm(pool.imap(calculate_mass_force, input_data), total=len(processed_data)))

    # Collect stats
    avg_metrics = ["average_force", "average_mass", "average_distance",
                   "average_question_bleu", "average_context_bleu",
                   "average_question_distance", "average_context_distance"]
    
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

    # avg_forces = [r["average_force"] for r in results]

    statistics["num_questions"] = len(results)
    print("📂 Saving results...")

    with open(os.path.join(output_dir, "-scores.json"), "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    # statistics = {
    #     "mean": float(np.mean(avg_forces)),
    #     "std": float(np.std(avg_forces)),
    #     "max": float(np.max(avg_forces)),
    #     "min": float(np.min(avg_forces)),
    #     "num_questions": len(results)
    # }

    with open(os.path.join(output_dir, "-statistics.json"), "w") as f:
        json.dump(statistics, f, indent=4, cls=NumpyEncoder)

    print("✅ Completed:")
    print(f"- Sub-question results written to: {os.path.join(output_dir, '-scores.json')}")
    print(f"- Statistics written to: {os.path.join(output_dir, '-statistics.json')}")
