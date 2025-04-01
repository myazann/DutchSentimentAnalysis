import argparse
import os
import json
import subprocess
from datasets import load_dataset, Dataset, DatasetDict
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, BitsAndBytesConfig, DataCollatorWithPadding, AutoTokenizer
from transformers.integrations import WandbCallback, TensorBoardCallback
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
import logging
import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from helpers import get_emotion_labels, get_translate_prompt, get_classifier_prompt, tokenize_function, prepare_data, compute_metrics
from models import LLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="ROBBERT-V2-EMOTION", type=str)
parser.add_argument("-lg", "--language", default="nl", type=str)
parser.add_argument("-wp", "--wandb_project", default="dutch-sentiment", type=str)
parser.add_argument("-wn", "--wandb_name", default="robbert-emotion", type=str)
parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning for compatible models.")
args = parser.parse_args()

learning_rate = 2e-5
weight_decay = 0.01
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1.0
gradient_accumulation_steps = 1

quantization_config = None
lora_config = None

translator_model_name = "GPT-4o-mini"
translator = LLM(translator_model_name, default_prompt=get_translate_prompt())

stored_translations = {}
if args.language == "nl":
    translations_file = os.path.join("files", f"translations_{translator_model_name}_{args.language}.json")
    if os.path.exists(translations_file):
        with open(translations_file, "r") as f:
            stored_translations = json.load(f)
    else:
        raise Exception("Prepare translations beforehand!")

dataset = load_dataset("li2017dailydialog/daily_dialog")

dataset = prepare_data(dataset, args, stored_translations, translator)
dataset = DatasetDict({
    split: Dataset.from_list(examples) 
    for split, examples in dataset.items()
})

num_labels = len(set(dataset["train"]["label"]))

model_params = {
    "num_labels": num_labels,
    "ignore_mismatched_sizes": True
}

if "LLAMA" in args.model or "FIETJE" in args.model:
    cuda_available = torch.cuda.is_available()
    model = LLM(args.model, model_params=model_params, use_lora=args.use_lora)
    
    if model.tokenizer.pad_token is None:
        if model.tokenizer.eos_token is not None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
            print(f"Using EOS token '{model.tokenizer.eos_token}' as padding token")
        else:
            model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added [PAD] token as padding token")
    
    print(f"Padding token: {model.tokenizer.pad_token}, ID: {model.tokenizer.pad_token_id}")
    
    # Determine quantization based on use_lora setting and CUDA availability
    quantization_config = None
    if args.use_lora and cuda_available:
        print("Using 4-bit quantization with LoRA")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        print("Not using quantization")

    try:
        # First try loading with CUDA and quantization if applicable
        print(f"Loading model from {model.repo_id}")
        classification_model = AutoModelForSequenceClassification.from_pretrained(
            model.repo_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            problem_type="single_label_classification",
            torch_dtype=torch.bfloat16 if (cuda_available and torch.cuda.is_bf16_supported()) else torch.float32,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map="auto" if cuda_available else None  # Explicitly use device_map="auto" for GPU
        )

        # Apply LoRA adapters if requested and model is compatible
        if args.use_lora and ("LLAMA" in args.model or "FIETJE" in args.model):
            print("Applying LoRA adapters as requested")
            # Prepare the model for k-bit training *if* quantization is enabled
            if quantization_config is not None:
                print("Preparing model for k-bit training before applying LoRA")
                classification_model = prepare_model_for_kbit_training(
                    classification_model,
                    use_gradient_checkpointing=True # Recommended with LoRA
                )

            # Configure LoRA
            lora_config = LoraConfig(
                r=16,  # Rank dimension
                lora_alpha=64,  # Alpha parameter for LoRA scaling
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type=TaskType.SEQ_CLS, # Specify task type for sequence classification
                lora_dropout=0.1, # Add dropout for regularization
                bias="none"
            )

            # Get PEFT model with LoRA configuration
            classification_model = get_peft_model(classification_model, lora_config)
            classification_model.print_trainable_parameters()

            # Explicitly move model to GPU and verify device (redundant with device_map='auto' but good check)
            if cuda_available:
                device = torch.device("cuda")
                classification_model = classification_model.to(device)
                # Check if model is actually on GPU
                print(f"Model device check after LoRA - Is model on CUDA: {next(classification_model.parameters()).is_cuda}")
                print(f"Model device after LoRA: {next(classification_model.parameters()).device}")
        else:
             print("LoRA not enabled or model not compatible.")

        # Set padding token in the model config
        if classification_model.config.pad_token_id is None:
            if model.tokenizer.pad_token_id is not None:
                classification_model.config.pad_token_id = model.tokenizer.pad_token_id
            else:
                # Use EOS token as padding token if no padding token is defined
                classification_model.config.pad_token_id = classification_model.config.eos_token_id
        
        if hasattr(classification_model, "classifier"):
            classification_model.classifier.weight.data.normal_(mean=0.0, std=0.02)
            classification_model.classifier.bias.data.zero_()
            print("Initialized classification head with better weights")
        
        # Replace the model's internal model with our classification model
        model.model = classification_model
        
        # Resize embeddings if needed
        if hasattr(model.model, "resize_token_embeddings") and model.tokenizer is not None:
            model.model.resize_token_embeddings(len(model.tokenizer))
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error: {e}")
            print("Falling back to CPU for model loading")
            # Try loading on CPU without quantization (quantization requires CUDA)
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
elif "BERT" in args.model or "ROBBERT" in args.model or "ROBERTA" in args.model:
    # For BERT-based models, we'll use the direct approach since these models
    # are natively designed for classification, not generative tasks
    cuda_available = torch.cuda.is_available()
    
    print(f"Loading BERT-based model: {args.model}")
    
    # Initialize the model directly with AutoModelForSequenceClassification 
    # instead of going through LLM class to avoid tokenization issues
    repo_id = LLM.get_cfg()[args.model].get("repo_id")
    print(f"Using repo_id: {repo_id}")
    
    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    
    # Load the model
    classification_model = AutoModelForSequenceClassification.from_pretrained(
        repo_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification"
    )
    
    # Move to GPU if available
    if cuda_available:
        classification_model = classification_model.to(torch.device("cuda"))
    
    # Create a simplified LLM wrapper with just what we need
    model = LLM(args.model, model_params=model_params)
    model.tokenizer = tokenizer
    model.model = classification_model
else:
    model = LLM(args.model, model_params=model_params, use_lora=args.use_lora)

logger.info(f"Number of unique labels in dataset: {num_labels}")

# Create cache directory for tokenized datasets
cache_dir = os.path.join("cache", args.model.replace("/", "_"))
os.makedirs(cache_dir, exist_ok=True)

logger.info("Tokenizing datasets (this may take a while)...")
num_proc = min(os.cpu_count() // 2, 4)  # Use half of available CPUs but max 4
tokenized_datasets = dataset.map(
    lambda examples: tokenize_function(args, model, examples), 
    batched=True,
    batch_size=1024,  
    num_proc=num_proc,  
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
if "LLAMA" in args.model or "FIETJE" in args.model:
    # Check if we're using quantization to determine batch size
    if args.use_lora and cuda_available:
        batch_size = 128 
    else:
        batch_size = 16
else:
    batch_size = 64

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
    fp16=args.use_lora and cuda_available,  # Enable fp16 for quantized models
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not (args.use_lora and cuda_available),  # Don't use bf16 with 4-bit quantization
    logging_steps=100,  
    report_to="wandb",
    warmup_steps=warmup_steps,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_grad_norm=max_grad_norm,
    gradient_checkpointing=args.use_lora, # Enable GC only if LoRA is used
    lr_scheduler_type="cosine",  
    dataloader_num_workers=num_proc,  
    dataloader_pin_memory=True,  
    optim="paged_adamw_8bit" if (args.use_lora and cuda_available) else "adamw_torch",  # Use 8-bit optimizer with quantization
    run_name=f"{args.model.split('/')[-1]}-{args.language}-emotion-{wandb_name}",
    no_cuda=False,
    ddp_find_unused_parameters=False, 
    group_by_length=True,  
    length_column_name="length",  
    remove_unused_columns=True,  
)

# Ensure the output directory exists
os.makedirs(training_args.output_dir, exist_ok=True)

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
            gpu_utilization = 0
            
            # Try to get GPU utilization from nvidia-smi
            try:
                nvidia_smi_output = subprocess.check_output(
                    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", 
                    shell=True
                ).decode().strip()
                gpu_utilization = float(nvidia_smi_output)
            except:
                pass
                
            # Use the exact global step for logging
            wandb.log({
                "gpu_memory_allocated_gb": memory_allocated,
                "gpu_memory_reserved_gb": memory_reserved,
                "gpu_utilization": gpu_utilization
            }, step=state.global_step)
            
            # Print GPU stats to console
            print(f"Step {state.global_step}: GPU Memory: {memory_allocated:.2f}GB allocated, "
                  f"{memory_reserved:.2f}GB reserved, Utilization: {gpu_utilization}%")

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

# Create appropriate data collator based on whether we're using quantization
if ("LLAMA" in args.model or "FIETJE" in args.model) and args.use_lora and cuda_available:
    print("Using DataCollatorWithPadding with 4-bit quantization compatibility")
    # For quantized models, we need to ensure we don't convert tensors to float16/bfloat16
    # as this would break the quantization
    data_collator = DataCollatorWithPadding(
        tokenizer=model.tokenizer, 
        padding="longest",
        return_tensors="pt"
    )
else:
    data_collator = DataCollatorWithPadding(
        tokenizer=model.tokenizer, 
        padding="longest"
    )

trainer = Trainer(
    model=model.model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator, 
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=10),
        WandbCallback(),
        LearningRateLoggerCallback(),
        GPUStatsCallback(),
        CheckpointSafetyCallback()
    ]
)

try:
    # Before training, ensure the output directory exists
    os.makedirs(training_args.output_dir, exist_ok=True)

    # --- Log additional hyperparameters to wandb ---
    if trainer.is_local_process_zero(): # Ensure logging happens only on main process
        additional_hyperparameters = {
            "script_args": vars(args),
            "translator_model_name": translator_model_name,
            "calculated_batch_size": batch_size,
            "num_unique_labels": num_labels,
            "quantization_config": quantization_config.to_dict() if quantization_config else None,
            "uses_lora_arg": args.use_lora, # Log the command line arg
            "lora_config": lora_config.to_dict() if lora_config else None,
            "hardcoded_lr": learning_rate,
            "hardcoded_wd": weight_decay,
            "hardcoded_warmup_ratio": warmup_ratio,
            "hardcoded_num_epochs": num_epochs,
            "hardcoded_max_grad_norm": max_grad_norm,
            "hardcoded_grad_accum": gradient_accumulation_steps
        }
        # Check if wandb run exists (initialized by Trainer/WandbCallback)
        if wandb.run:
            wandb.config.update(additional_hyperparameters, allow_val_change=True)
            logger.info(f"Updated wandb config with additional hyperparameters: {list(additional_hyperparameters.keys())}")
        else:
            logger.warning("Wandb run not initialized, cannot log additional hyperparameters.")

    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    final_model_path = os.path.join(model.finetune_path, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    
    # For quantized models with LoRA, save adapters separately
    if ("LLAMA" in args.model or "FIETJE" in args.model) and args.use_lora and cuda_available:
        logger.info("Saving LoRA adapters for quantized model...")
        # Save only the LoRA adapters
        model.model.save_pretrained(final_model_path)
    else:
        # Save the full model for non-quantized models
        trainer.save_model(final_model_path)
    
    # Always save the tokenizer
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
