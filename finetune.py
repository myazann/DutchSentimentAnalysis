import argparse
import os
import json
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from transformers.integrations import WandbCallback, TensorBoardCallback
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb

from helpers import get_emotion_labels, get_translate_prompt
from models import LLM

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-lg", "--language", default="nl", type=str)
parser.add_argument("--wandb_project", default="dutch-sentiment", type=str)
parser.add_argument("--wandb_name", default="robbert-emotion", type=str)
args = parser.parse_args()

translator_model_name = "GPT-4o-mini"
translator = LLM(translator_model_name, default_prompt=get_translate_prompt())

if args.language == "nl":
    translations_file = os.path.join("files", f"translations_{translator_model_name}_{args.language}.json")
    if os.path.exists(translations_file):
        with open(translations_file, "r") as f:
            stored_translations = json.load(f)
    else:
        raise Exception("Prepare translations beforehand!")

dataset = load_dataset("li2017dailydialog/daily_dialog")

def prepare_data(dataset):

    prepped_dataset = {}
    for split in ["train", "validation"]:
        prepped_dataset[split] = []
        dialogs = dataset[split]["dialog"]
        labels = dataset[split]["emotion"]
        for i in range(len(dialogs)):
            for j in range(len(dialogs[i])):

                text = dialogs[i][j]
                if stored_translations[split].get(text, None) is not None:
                    translated_text = stored_translations[split][text]
                else:
                    print("Couldn't find translation!")
                    print(text)
                    translated_text = translator.generate(prompt_params={"text": text})
                    stored_translations[split][text] = translated_text
                
                prepped_dataset[split].append({
                    "text": translated_text,
                    "label": labels[i][j]
                })
    return prepped_dataset

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

dataset = prepare_data(dataset)
dataset = DatasetDict({
    split: Dataset.from_list(examples) 
    for split, examples in dataset.items()
})

tokenizer = AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
tokenized_datasets = dataset.map(tokenize_function, batched=True, disable_progress_bar=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Get the number of unique labels
num_labels = len(set(dataset["train"]["label"]))
logger.info(f"Number of unique labels in dataset: {num_labels}")

model = AutoModelForSequenceClassification.from_pretrained(
    "pdelobelle/robbert-v2-dutch-base", 
    num_labels=num_labels
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Initialize wandb
wandb.init(
    project=args.wandb_project,
    name=args.wandb_name,
    config={
        "learning_rate": 2e-5,
        "batch_size": 32,
        "model": "pdelobelle/robbert-v2-dutch-base",
        "epochs": 3,
        "language": args.language
    }
)

training_args = TrainingArguments(
    output_dir="./finetune",
    evaluation_strategy="steps",
    eval_steps=500,              
    logging_dir="./logs",
    logging_steps=10,           
    save_strategy="steps",
    save_steps=1000,            
    learning_rate=2e-5,
    per_device_train_batch_size=64,  
    per_device_eval_batch_size=64,   
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="wandb",          
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        WandbCallback(),
        TensorBoardCallback()
    ]
)
# Log dataset sizes
logger.info(f"Training dataset size: {len(tokenized_datasets['train'])}")
logger.info(f"Validation dataset size: {len(tokenized_datasets['validation'])}")

# Start training
logger.info("Starting training...")
train_result = trainer.train()

# Log final results
logger.info("Training completed!")
logger.info(f"Final train metrics: {train_result.metrics}")

# Evaluate on validation set
logger.info("Running final evaluation...")
eval_results = trainer.evaluate()
logger.info(f"Final evaluation metrics: {eval_results}")

# Save final model
trainer.save_model("./finetune/final-model")
logger.info("Model saved to ./finetune/final-model")

# After training completes
wandb.finish()