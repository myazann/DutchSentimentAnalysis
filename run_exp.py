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
# Update model names to remove FINETUNED suffix where needed
llm_list = ["XLM-ROBERTA-BASE", "MULTILINGUAL-BERT", "LLAMA-3.2-3B", "FIETJE-2-CHAT", "LLAMA-3.2-1B", "ROBBERT-V2-EMOTION", "QWEN-2.5-72B-GGUF", "LLAMA-3.3-70B-GGUF"]
# Keep track of which models need finetuned=True
finetuned_models = ["XLM-ROBERTA-BASE", "MULTILINGUAL-BERT", "LLAMA-3.2-3B", "FIETJE-2-CHAT", "LLAMA-3.2-1B", "ROBBERT-V2-EMOTION"]

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
    # Skip finetuned models when using few-shot
    if args.few_k > 0 and model_name in finetuned_models:
        continue
    
    # Determine if we should use the finetuned version    
    use_finetuned = model_name in finetuned_models
    
    # Use the original names for experiment tracking
    exp_model_name = f"{model_name}-FINETUNED" if use_finetuned else model_name
    exp_name = f"preds_{exp_model_name}_{args.language}_k({args.few_k})_p({args.proportional})"
    print(f"\n{exp_model_name}")
  
    all_res = []

    if os.path.exists(os.path.join("files", f"{exp_name}.json")):
        with open(os.path.join("files", f"{exp_name}.json"), "r") as f:
            all_res = json.load(f)
        if len(all_res) == len(daily_dialog_test):
            continue

    if use_finetuned:
        if "ROBBERT" in model_name:
            classifier = LLM(model_name, model_params={"num_labels": 7, "ignore_mismatched_sizes": True}, finetuned=True)
        else:
            is_large_model = False
            
            try:
                model_cfg = LLM.get_cfg()[model_name]
                min_gpu_ram = model_cfg.get("min_GPU_RAM")
                
                if min_gpu_ram is not None:
                    try:
                        min_gpu_ram = int(min_gpu_ram)
                        is_large_model = min_gpu_ram >= 10
                        print(f"Model min_GPU_RAM: {min_gpu_ram}GB, Using quantization: {is_large_model}")
                    except (ValueError, TypeError):
                        is_large_model = False
                        print("Could not determine min_GPU_RAM, not using quantization")
                else:
                    print("No min_GPU_RAM specified in config, not using quantization")
            except KeyError:
                print(f"Model {model_name} not found in config, not using quantization")
                is_large_model = False
            
            if is_large_model and torch.cuda.is_available():
                base_model = LLM(model_name, default_prompt=get_classifier_prompt(args.language), finetuned=True)

                # Check if the model is a LoRA adapter by looking for adapter_config.json
                finetune_path = os.path.join("finetune", base_model.repo_id.split("/")[-1], "final_model")
                adapter_config_path = os.path.join(finetune_path, "adapter_config.json")
                is_lora_adapter = os.path.exists(adapter_config_path)

                if is_lora_adapter:
                    print(f"Loading as LoRA adapter: {model_name}")
                    # Setup quantization config
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    
                    quantized_model = AutoModelForSequenceClassification.from_pretrained(
                        base_model.repo_id,
                        num_labels=7,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    )
                    
                    classifier_model = PeftModel.from_pretrained(
                        quantized_model,
                        finetune_path,
                        is_trainable=False  # Set to False for inference
                    )
                    
                    base_model.model = classifier_model
                    classifier = base_model
                else:
                    classifier = LLM(model_name, default_prompt=get_classifier_prompt(args.language), model_params={"num_labels": 7, "ignore_mismatched_sizes": True}, finetuned=True)
            else:
                classifier = LLM(model_name, default_prompt=get_classifier_prompt(args.language), model_params={"num_labels": 7, "ignore_mismatched_sizes": True}, finetuned=True)
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
                if use_finetuned and "instruct" not in model_name:
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
