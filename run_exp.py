import json
import argparse
import subprocess
import os
import sys
import copy
from datasets import load_dataset
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

from models import LLM
from helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--few_k", default=0, type=int)
parser.add_argument("-lg", "--language", default="nl", type=str)
parser.add_argument("-p", "--proportional", type=bool, default=False)
parser.add_argument("-ob", "--openai_batch", default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

if args.few_k == 0:
    args.proportional = True

# llm_list = ["GPT-4o-mini"]
translator = None
llm_list = ["LLAMA-1B-FINETUNED", "LLAMA-3B-FINETUNED", "ROBBERT-V2-EMOTION-FINETUNED", "QWEN-2.5-72B-GGUF", "LLAMA-3.3-70B-GGUF", "QWEN-2.5-14B-GGUF"]

if args.language == "nl":
    translator_model_name = "GPT-4o-mini"
    translator = LLM(translator_model_name, default_prompt=get_translate_prompt())

daily_dialog_test = load_dataset("li2017dailydialog/daily_dialog")["test"]
few_shot_examples = ""

if args.few_k > 0:
    few_shot_file = os.path.join("files", f"few_shot_examples_{args.language}_k({args.few_k})_p({args.proportional}).txt")
    if os.path.exists(few_shot_file):
        with open(few_shot_file, "r") as f:
            few_shot_examples = f.read()
    else:
        daily_dialog_train = load_dataset("li2017dailydialog/daily_dialog")["train"]
        few_shot_examples = get_few_shot_samples(daily_dialog_train, args, translator)

        with open(few_shot_file, "w") as f:
            f.write(few_shot_examples)

if args.language == "nl":
    translations_file = os.path.join("files", f"translations_{translator_model_name}_{args.language}.json")
    if os.path.exists(translations_file):
        with open(translations_file, "r") as f:
            stored_translations = json.load(f)
    else:
        stored_translations = {"test": {}}

emotion_labels_inverse = get_emotion_labels_inverse(args.language)
print(f"Language: {args.language}, Few shot k: {args.few_k}, Proportional: {args.proportional}")

for model_name in llm_list:
    if args.few_k > 0 and "FINETUNED" in model_name:
        continue
        
    exp_name = f"preds_{model_name}_{args.language}_k({args.few_k})_p({args.proportional})"
    print(f"\n{model_name}")
  
    all_res = []

    if os.path.exists(os.path.join("files", f"{exp_name}.json")):
        with open(os.path.join("files", f"{exp_name}.json"), "r") as f:
            all_res = json.load(f)
        if len(all_res) == len(daily_dialog_test):
            continue

    if "FINETUNED" in model_name:
        if "ROBBERT" in model_name:
            classifier = LLM(model_name, model_params={"num_labels": 7, "ignore_mismatched_sizes": True})
        else:
            # Check if this is a quantized model (models larger than 1B)
            model_size_str = model_name.split("-")[-2]  # Get size part (e.g., "3B" from "LLAMA-3B-FINETUNED")
            is_large_model = False
            if model_size_str.endswith("B"):
                try:
                    model_size = float(model_size_str[:-1])
                    is_large_model = model_size > 1.0
                    print(f"Model size: {model_size}B, Using quantization: {is_large_model}")
                except ValueError:
                    # If we can't parse the size, assume it's not a large model
                    pass
            
            if is_large_model and torch.cuda.is_available():
                print(f"Loading quantized model with LoRA adapters: {model_name}")
                # Initialize base LLM for tokenizer and config
                base_model = LLM(model_name, default_prompt=get_classifier_prompt(args.language))
                
                # Setup quantization config
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                
                # Load the base model with quantization
                base_model_path = base_model.repo_id
                adapter_path = os.path.join("finetune", base_model_path.split("/")[-1], "final_model")
                
                # Load the quantized base model
                quantized_model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_path,
                    num_labels=7,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
                
                # Load the LoRA adapters
                print(f"Loading adapters from: {adapter_path}")
                classifier_model = PeftModel.from_pretrained(
                    quantized_model,
                    adapter_path,
                    is_trainable=False  # Set to False for inference
                )
                
                # Replace the model in the LLM instance
                base_model.model = classifier_model
                classifier = base_model
            else:
                # For smaller models, load normally
                classifier = LLM(model_name, default_prompt=get_classifier_prompt(args.language), model_params={"num_labels": 7, "ignore_mismatched_sizes": True})
    else:
        classifier = LLM(model_name, default_prompt=get_classifier_prompt(args.language))

    subprocess.run("gpustat")
    cont_idx = copy.copy(len(all_res))
    print(f"Continuing from sample {cont_idx}")

    all_res = []
    for _ in range(len(daily_dialog_test) - len(all_res)):

        turn_preds = []
        turn_dialog = daily_dialog_test["dialog"][cont_idx]

        for turn in turn_dialog:    

            if args.language == "nl":
                if turn in stored_translations["test"]:
                    text = stored_translations["test"][turn]
                else:
                    translated_text = translator.generate(prompt_params={"text": turn})
                    stored_translations["test"][turn] = translated_text
                    text = stored_translations["test"][turn]
                    with open(translations_file, "w") as f:
                        json.dump(stored_translations, f, indent=4)
            else:
                text = turn
            try:
                if classifier.provider == "FINETUNED":
                    pred = classifier.generate(prompt_params={"text": text, "language": args.language})
                else:
                    pred = classifier.generate(prompt_params={"text": text, "few_shot_examples": few_shot_examples})
                    pred = emotion_labels_inverse[pred]
                turn_preds.append(pred)
            except Exception as e:
                print(e)
                turn_preds.append(-1)

        all_res.append(turn_preds)

        with open(os.path.join("files", f"{exp_name}.json"), "w") as f:
            json.dump(all_res, f)

        cont_idx += 1
        sys.stdout.flush()
