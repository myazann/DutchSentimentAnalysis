import os 
import pandas as pd
import json
from datasets import load_dataset

gts = load_dataset("li2017dailydialog/daily_dialog")["test"]["emotion"]
flattened_gts = [item for sublist in gts for item in sublist]

pred_files = [f for f in os.listdir("files") if f.startswith("preds")]
for file in pred_files:

    with open(os.path.join("files", file), "r") as f:
        preds = json.load(f)

    flattened_preds = [item for sublist in preds for item in sublist]
    accuracy = 0

    if len(flattened_gts) != len(flattened_preds):
        continue

    print(file)

    # Calculate accuracy
    for gt, pred in zip(flattened_gts, flattened_preds):
        if pred == gt:
            accuracy += 1
    
    accuracy_score = round(accuracy/len(flattened_gts), 3)
    f1_scores = []
    classes = set(flattened_gts)
    
    for cls in classes:
        true_pos = sum(1 for g, p in zip(flattened_gts, flattened_preds) if g == cls and p == cls)
        false_pos = sum(1 for g, p in zip(flattened_gts, flattened_preds) if g != cls and p == cls)
        false_neg = sum(1 for g, p in zip(flattened_gts, flattened_preds) if g == cls and p != cls)
        
        class_total = sum(1 for g in flattened_gts if g == cls)
        class_accuracy = round(true_pos / class_total, 3) if class_total > 0 else 0
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
        print(f"Class {cls}:")
        print(f"  Accuracy: {class_accuracy}")
        print(f"  F1 Score: {round(f1, 3)}")
    
    macro_f1 = round(sum(f1_scores) / len(f1_scores), 3)
    
    print(f"\nOverall Accuracy: {accuracy_score}")
    print(f"Macro F1: {macro_f1}")