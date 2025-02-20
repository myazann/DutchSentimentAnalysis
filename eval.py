import os 
import json
from datasets import load_dataset

gts = load_dataset("li2017dailydialog/daily_dialog")["test"]["emotion"]
flattened_gts = [item for sublist in gts for item in sublist]

pred_files = [f for f in os.listdir("files") if f.startswith("preds")]
for file in pred_files:

    with open(os.path.join("files", file), "r") as f:
        preds = json.load(f)

    if len(gts) != len(preds):
        continue

    print(file)
    flattened_preds = [item for sublist in preds for item in sublist]
    accuracy = 0

    for gt, pred in zip(flattened_gts, flattened_preds):
        if pred == gt:
            accuracy += 1
    
    print(round(accuracy/len(flattened_gts), 3))

