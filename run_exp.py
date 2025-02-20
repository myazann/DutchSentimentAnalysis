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
parser.add_argument("-k", "--few_k", default=1, type=int)
parser.add_argument("-lg", "--language", default="nl", type=str)
parser.add_argument("-ob", "--openai_batch", default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

llm_list = ["LLAMA-3.3-70B", "QWEN-2.5-14B-1M", "QWEN-2.5-7B-1M"]
translator_model_name = "GPT-4o-mini"
translator = LLM(translator_model_name, default_prompt=get_translate_prompt())

daily_dialog_test = load_dataset("li2017dailydialog/daily_dialog")["test"]

few_shot_examples = ""

if args.few_k > 0:
    few_shot_file = os.path.join("files", f"few_shot_examples_{args.language}_k({args.few_k}).txt")
    if os.path.exists(few_shot_file):
        with open(few_shot_file, "r") as f:
            few_shot_examples = f.read()
    else:
        daily_dialog_train = load_dataset("li2017dailydialog/daily_dialog")["train"]
        few_shot_examples = get_few_shot_samples(daily_dialog_train, translator, args.few_k)
        with open(few_shot_file, "w") as f:
            f.write(few_shot_examples)

translations_file = os.path.join("files", f"translations_{translator_model_name}_{args.language}.json")
if os.path.exists(translations_file):
    with open(translations_file, "r") as f:
        stored_translations = json.load(f)
else:
    stored_translations = {"translations": {}}

emotion_labels_inverse = get_emotion_labels_inverse()

for llm in llm_list:
    exp_name = f"preds_{llm}_{args.language}_k({args.few_k})"
    print(exp_name)
    all_res = []

    if os.path.exists(os.path.join("files", f"{exp_name}.json")):
        with open(os.path.join("files", f"{exp_name}.json"), "r") as f:
            all_res = json.load(f)
        if len(all_res) == len(daily_dialog_test):
            continue

    if args.language == "nl":
        classifier = LLM(llm, default_prompt=get_classifier_prompt_nl())

    subprocess.run("gpustat")
    cont_idx = copy.copy(len(all_res))
    print(f"Continuing from sample {cont_idx}")

    for _ in range(len(daily_dialog_test) - len(all_res)):
        try:
            original_text = str(daily_dialog_test["dialog"][cont_idx])

            if original_text in stored_translations["translations"]:
                translated_text = stored_translations["translations"][original_text]
            else:
                translated_text = translator.generate(prompt_params={"text": original_text})
                stored_translations["translations"][original_text] = translated_text
                with open(translations_file, "w") as f:
                    json.dump(stored_translations, f, indent=4)

            pred = classifier.generate(prompt_params={"text": translated_text, "few_shot_examples": few_shot_examples})
            all_res.append([emotion_labels_inverse[p] for p in eval(pred)])
        except Exception as e:
            print(e)
            print(pred)
            all_res.append(pred)

        with open(os.path.join("files", f"{exp_name}.json"), "w") as f:
            json.dump(all_res, f)

        cont_idx += 1
        sys.stdout.flush()
