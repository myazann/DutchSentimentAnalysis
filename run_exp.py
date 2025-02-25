import json
import argparse
import subprocess
import os
import sys
import copy
from datasets import load_dataset

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
llm_list = ["ROBBERT-EMOTION-FINETUNED", "QWEN-2.5-72B-GGUF", "LLAMA-3.3-70B-GGUF", "QWEN-2.5-14B-GGUF"]

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
        classifier = LLM(model_name)
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
                    pred = classifier.generate(prompt_params={"text": text})
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
