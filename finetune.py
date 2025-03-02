import argparse
import os
import json
from datasets import load_dataset, Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
from transformers.integrations import WandbCallback, TensorBoardCallback
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import torch
from torch.utils.data import DataLoader

from helpers import get_emotion_labels, get_translate_prompt
from models import LLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="ROBBERT-V2-EMOTION-FINETUNED", type=str)
parser.add_argument("-lg", "--language", default="nl", type=str)
parser.add_argument("--wandb_project", default="dutch-sentiment", type=str)
parser.add_argument("-wn", "--wandb_name", default="robbert-emotion", type=str)
args = parser.parse_args()

learning_rate = 2e-5
weight_decay = 0.01
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1.0
gradient_accumulation_steps = 4

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

def tokenize_function(model, examples):
    # For LLaMA models, we need special handling
    if "LLAMA" in args.model:
        # Add special tokens for better instruction tuning
        emotion_labels_dict = get_emotion_labels(args.language)
        emotion_labels_list = list(emotion_labels_dict.values())
        texts = []
        for text in examples['text']:
            # Format as an instruction with context about the emotion classification task
            prompt = f"<s>[INST] Classify the emotion in this Dutch text into one of these categories: {', '.join(emotion_labels_list)}.\n\nText: {text} [/INST]"
            texts.append(prompt)
        
        # Use a shorter max_length for LLAMA models to reduce memory usage
        return model.tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=256,  # Reduced from 384 to 256 to save memory
            return_tensors="pt"
        )
    else:
        return model.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

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

dataset = prepare_data(dataset)
dataset = DatasetDict({
    split: Dataset.from_list(examples) 
    for split, examples in dataset.items()
})

num_labels = len(set(dataset["train"]["label"]))

# Initialize the model with appropriate parameters for classification
model_params = {
    "num_labels": num_labels,
    "ignore_mismatched_sizes": True
}

# For LLaMA models, we need to add a classification head
if "LLAMA" in args.model:
    print(f"Using LLaMA model: {args.model}")
    # Use AutoModelForSequenceClassification directly for LLaMA models
    from transformers import AutoModelForSequenceClassification
    import torch
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            # Try to free up CUDA memory
            torch.cuda.empty_cache()
            print("Using CUDA for training")
        except Exception as e:
            print(f"CUDA error: {e}")
            print("Falling back to CPU")
            cuda_available = False
    else:
        print("CUDA not available, using CPU")
    
    # Initialize LLM just for the tokenizer
    model = LLM(args.model, model_params=model_params)
    
    # Make sure the tokenizer has a padding token
    if model.tokenizer.pad_token is None:
        if model.tokenizer.eos_token is not None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
            print(f"Using EOS token '{model.tokenizer.eos_token}' as padding token")
        else:
            model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added [PAD] token as padding token")
    
    print(f"Padding token: {model.tokenizer.pad_token}, ID: {model.tokenizer.pad_token_id}")
    
    # Then manually create the classification model with better initialization
    try:
        # First try loading with CUDA
        classification_model = AutoModelForSequenceClassification.from_pretrained(
            model.repo_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            problem_type="single_label_classification",
            torch_dtype=torch.bfloat16 if (cuda_available and torch.cuda.is_bf16_supported()) else torch.float32,  # Use bfloat16 when possible
            low_cpu_mem_usage=True
        )
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error: {e}")
            print("Falling back to CPU for model loading")
            # Try loading on CPU
            classification_model = AutoModelForSequenceClassification.from_pretrained(
                model.repo_id,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                problem_type="single_label_classification",
                device_map="cpu",
                low_cpu_mem_usage=True
            )
        else:
            raise e
    
    # Set padding token in the model config
    if classification_model.config.pad_token_id is None:
        if model.tokenizer.pad_token_id is not None:
            classification_model.config.pad_token_id = model.tokenizer.pad_token_id
        else:
            # Use EOS token as padding token if no padding token is defined
            classification_model.config.pad_token_id = classification_model.config.eos_token_id
    
    # Initialize classification head with better weights for stable training
    if hasattr(classification_model, "classifier"):
        # Initialize the classification head with small values
        classification_model.classifier.weight.data.normal_(mean=0.0, std=0.02)
        classification_model.classifier.bias.data.zero_()
        print("Initialized classification head with better weights")
    
    # Replace the model's internal model with our classification model
    model.model = classification_model
    
    # Resize embeddings if needed
    if hasattr(model.model, "resize_token_embeddings") and model.tokenizer is not None:
        model.model.resize_token_embeddings(len(model.tokenizer))
else:
    model = LLM(args.model, model_params=model_params)

logger.info(f"Number of unique labels in dataset: {num_labels}")

# Create cache directory for tokenized datasets
cache_dir = os.path.join("cache", args.model.replace("/", "_"))
os.makedirs(cache_dir, exist_ok=True)

# Skip using cache due to PyTorch 2.6 compatibility issues
logger.info("Tokenizing datasets (this may take a while)...")
# Use more workers for faster processing
num_proc = min(os.cpu_count() // 2, 4)  # Use half of available CPUs but max 4
tokenized_datasets = dataset.map(
    lambda examples: tokenize_function(model, examples), 
    batched=True,
    batch_size=1024,  # Larger batch size for efficiency
    num_proc=num_proc,  # Parallel processing
    desc="Tokenizing datasets"
)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Save tokenized datasets to cache (for future use if needed)
cache_file = os.path.join(cache_dir, f"tokenized_datasets_{args.language}.pt")
logger.info(f"Saving tokenized datasets to cache: {cache_file}")
try:
    torch.save(tokenized_datasets, cache_file)
except Exception as e:
    logger.warning(f"Failed to save tokenized datasets to cache: {e}")
    logger.info("Continuing without saving cache...")

wandb_name = args.wandb_name if args.wandb_name else f"{args.model.split('/')[-1]}-{args.language}-emotion"
wandb.init(
    project=args.wandb_project,
    name=wandb_name,
    config={
        "model": args.model,
        "language": args.language,
        "num_labels": num_labels,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "num_epochs": num_epochs,
        "max_grad_norm": max_grad_norm,
        "gradient_accumulation_steps": gradient_accumulation_steps
    }
)

# Calculate total training steps for warmup
num_train_samples = len(tokenized_datasets["train"])

# Determine optimal batch size based on model
if "LLAMA" in args.model:
    batch_size = 8  # Slightly larger batch size for LLaMA
else:
    batch_size = 32

total_train_steps = (num_train_samples // (batch_size * gradient_accumulation_steps)) * num_epochs
warmup_steps = int(total_train_steps * warmup_ratio)

logger.info(f"Total training steps: {total_train_steps}, Warmup steps: {warmup_steps}")
logger.info(f"Using batch size: {batch_size}, Gradient accumulation steps: {gradient_accumulation_steps}")

training_args = TrainingArguments(
    output_dir=model.finetune_path,
    evaluation_strategy="steps",
    eval_steps=500,              
    logging_dir="./logs",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,   
    num_train_epochs=num_epochs,
    weight_decay=weight_decay,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    fp16=False,  # Disable fp16 to avoid gradient scaling issues
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # Use bf16 if available as it's more stable
    logging_steps=100,  
    report_to="wandb",
    warmup_steps=warmup_steps,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_grad_norm=max_grad_norm,
    gradient_checkpointing=True if "LLAMA" in args.model else False,
    lr_scheduler_type="cosine",  # Use cosine scheduler for better convergence
    dataloader_num_workers=num_proc,  # Use multiple workers for data loading
    dataloader_pin_memory=True,  # Pin memory for faster data transfer to GPU
    optim="adamw_torch",  # Use torch implementation for better performance
    run_name=f"{args.model.split('/')[-1]}-{args.language}-emotion-{wandb_name}",  # Add a unique run name
)

# Ensure the output directory exists
os.makedirs(model.finetune_path, exist_ok=True)

# Create a custom callback to log learning rate
class LearningRateLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs is not None:
            if 'learning_rate' in logs:
                # Use the exact global step for logging
                wandb.log({"learning_rate": logs["learning_rate"]}, step=state.global_step)

# Add a custom callback to track GPU memory usage
class GPUStatsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available() and state.is_local_process_zero:
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
            # Use the exact global step for logging
            wandb.log({
                "gpu_memory_allocated_gb": memory_allocated,
                "gpu_memory_reserved_gb": memory_reserved
            }, step=state.global_step)

# Add a custom callback to handle missing checkpoint directories
class CheckpointSafetyCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        """Ensure checkpoint directories exist before saving."""
        if state.is_local_process_zero and state.best_model_checkpoint is not None:
            # Make sure the best model checkpoint directory exists
            os.makedirs(state.best_model_checkpoint, exist_ok=True)
            
            # If we're about to save a new checkpoint, make sure its directory exists
            if control.should_save:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
        return control

# Create a custom data collator that handles padding efficiently
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer, padding="longest")

trainer = Trainer(
    model=model.model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator, 
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5),
        WandbCallback(),
        LearningRateLoggerCallback(),
        GPUStatsCallback(),
        CheckpointSafetyCallback()
    ]
)

try:
    # Before training, ensure the output directory exists
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    final_model_path = os.path.join(model.finetune_path, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_model(final_model_path)
    model.tokenizer.save_pretrained(final_model_path)
    
except FileNotFoundError as e:
    # Handle the specific error about missing checkpoint directories
    logger.error(f"Checkpoint directory error: {e}")
    logger.info("Attempting to recover by creating missing directories...")
    
    # Try to extract the missing path from the error message
    error_str = str(e)
    if "No such file or directory:" in error_str:
        missing_path = error_str.split("No such file or directory:")[-1].strip()
        try:
            # Create the missing directory
            os.makedirs(missing_path, exist_ok=True)
            logger.info(f"Created missing directory: {missing_path}")
            
            # Try to resume training
            logger.info("Resuming training...")
            trainer.train()
            
            # Save the final model
            final_model_path = os.path.join(model.finetune_path, "final_model")
            os.makedirs(final_model_path, exist_ok=True)
            trainer.save_model(final_model_path)
            model.tokenizer.save_pretrained(final_model_path)
        except Exception as inner_e:
            logger.error(f"Failed to recover: {inner_e}")
            raise
    else:
        # If we can't parse the error, re-raise it
        raise
except Exception as e:
    # Handle any other exceptions
    logger.error(f"Training error: {e}")
    raise

# Evaluate the model on the test set
test_results = trainer.evaluate(tokenized_datasets["validation"])
print(f"Final evaluation results: {test_results}")

# Close wandb
wandb.finish()
