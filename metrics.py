import csv
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def compute_scores(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [], "BERT-F1": []}
    
    # Compute ROUGE scores
    for pred, ref in zip(predictions, references):
        rouge_scores = scorer.score(pred, ref)
        scores["ROUGE-1"].append(rouge_scores["rouge1"].fmeasure)
        scores["ROUGE-2"].append(rouge_scores["rouge2"].fmeasure)
        scores["ROUGE-L"].append(rouge_scores["rougeL"].fmeasure)
    
    # Compute BERTScore F1
    P, R, F1 = bert_score(predictions, references, lang="en", rescale_with_baseline=True)
    scores["BERT-F1"].extend(F1.tolist())
    
    return {key: np.mean(value) for key, value in scores.items()}

def save_scores(scores, model_name, experiment_type, dataset_name):
    with open("rouge_results.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([model_name, experiment_type, dataset_name, scores["ROUGE-1"], scores["ROUGE-2"], scores["ROUGE-L"], scores["BERT-F1"]])