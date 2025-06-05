from nltk.corpus import stopwords
from collections import Counter
import nltk
 
STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(text: str) -> str:
    # tokens = nltk.word_tokenize(text)
    start_with_hash_tokens = start_with_hash(text)
    tokens = text.split()
    tokens = [t for t in tokens if t.lower() not in start_with_hash_tokens]
    text = " ".join(tokens)
    text = remove_punctuation(text)
    
    return text


def start_with_hash(text: str) -> bool:
    import re
    # Regular expression to match '#' followed by an integer or decimal number
    pattern = r"#\d+(\.\d+)?"
    # Find all matches
    matches = re.findall(pattern, text)
    # If you want the full match including the '#', use re.finditer
    full_matches = [match.group() for match in re.finditer(pattern, text)]

    return full_matches


def remove_punctuation(text: str) -> str:
    import string
    # Create a translation table that maps punctuation to None
    translator = str.maketrans('', '', string.punctuation)
    # Translate the text using the translation table
    return text.translate(translator)





def compute_bleu_precision(reference: str, candidate: str) -> float:
    """
    Computes BLEU-1 precision (unigram-level) between a candidate and reference.
    """
    print(f"[DEBUG] compute_bleu_precision: \nreference='{reference}', \ncandidate='{candidate}\n\n'")
    reference = remove_stopwords(reference)
    candidate = remove_stopwords(candidate)
    print(f"[DEBUG] After stopword removal: \nreference='{reference}', \ncandidate='{candidate}\n\n'")
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    
    print(f"[DEBUG] ref_counts={ref_counts}, \ncand_counts={cand_counts}\n\n")
    
    # Clipped count: min(count in candidate, count in reference)
    clipped = sum(min(cand_counts[w], ref_counts[w]) for w in cand_counts)
    
    total_cand_unigrams = sum(cand_counts.values())
    
    if total_cand_unigrams == 0:
        return 0.0
    print(f"[DEBUG] clipped={clipped}, \ntotal_cand_unigrams={total_cand_unigrams}\n\n")
    input("Press Enter to continue...")  # For debugging purposes, remove in production

    precision = clipped / total_cand_unigrams
    return precision

reference = "There is a state A where Storm of the Century was filmed based on the book by the author who wrote a short story featuring the author of This Is My God. What is the population of state A?"
candidate = "where was #2 storm of the century filmed"
print(compute_bleu_precision(reference, candidate))