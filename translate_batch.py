import openai
import os
import argparse
import json
from datasets import load_dataset

from models import LLM
from helpers import get_translate_prompt


def oai_get_or_create_file(client, filename):

    files = client.files.list()
    existing_file = next((file for file in files if file.filename == filename), None)

    if existing_file:
        print(f"File '{filename}' already exists. File ID: {existing_file.id}")
        return existing_file.id
    else:
        with open(filename, "rb") as file_data:
            new_file = client.files.create(
                file=file_data,
                purpose="batch"
            )
        print(f"File '{filename}' created. File ID: {new_file.id}")
        return new_file.id


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--split", default="train", type=str)
parser.add_argument("-lg", "--language", default="nl", type=str)
args = parser.parse_args()

daily_dialog = load_dataset("li2017dailydialog/daily_dialog")[args.split]["dialog"]
MAX_NEW_TOKENS = 128

llm = LLM("GPT-4o-mini")

# Split dialogs into two halves
half_length = len(daily_dialog) // 2
first_half = daily_dialog[:half_length]
second_half = daily_dialog[half_length:]

# Process first half
file_name_1 = os.path.join("files", f"batch_{llm.repo_id}_{args.split}_{args.language}_translations_1.jsonl")
for i, dialog in enumerate(first_half):
    for j, turn in enumerate(dialog):
        translate_prompt = get_translate_prompt()
        prompt = llm.format_prompt(translate_prompt, {"text": turn})
        with open(file_name_1, "a+") as file:
            json_line = json.dumps({"custom_id": f"{i}_{j}", "method": "POST", "url": "/v1/chat/completions",
                                  "body": {"model": llm.repo_id,
                                  "messages": prompt, "max_tokens": MAX_NEW_TOKENS}})
            file.write(json_line + '\n')

# Process second half            
file_name_2 = os.path.join("files", f"batch_{llm.repo_id}_{args.split}_{args.language}_translations_2.jsonl")
for i, dialog in enumerate(second_half):
    for j, turn in enumerate(dialog):
        translate_prompt = get_translate_prompt()
        prompt = llm.format_prompt(translate_prompt, {"text": turn})
        with open(file_name_2, "a+") as file:
            json_line = json.dumps({"custom_id": f"{i}_{j}", "method": "POST", "url": "/v1/chat/completions",
                                  "body": {"model": llm.repo_id,
                                  "messages": prompt, "max_tokens": MAX_NEW_TOKENS}})
            file.write(json_line + '\n')

# Create two separate batch requests
batch_input_file_id_1 = oai_get_or_create_file(llm.model, file_name_1)
batch_input_file_id_2 = oai_get_or_create_file(llm.model, file_name_2)

llm.model.batches.create(
    input_file_id=batch_input_file_id_1,
    endpoint="/v1/chat/completions", 
    completion_window="24h",
)

llm.model.batches.create(
    input_file_id=batch_input_file_id_2,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
