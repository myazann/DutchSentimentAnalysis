import argparse
import os
from datasets import load_dataset
import json

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--split", default="train", type=str)
parser.add_argument("-lg", "--language", default="nl", type=str)
args = parser.parse_args()

client = OpenAI()
model_name = "GPT-4o-mini"
file_name = f"batch_{model_name.lower()}_{args.split}_{args.language}_translations"
files = [f for f in os.listdir("files") if f.startswith(file_name)]

batches = client.batches.list()
files = [f for f in client.files.list() if f.filename in files]
daily_dialog = load_dataset("li2017dailydialog/daily_dialog")[args.split]["dialog"]

out_path = os.path.join("files", f"translations_{model_name}_{args.language}.json")

all_batch_res = []
for batch in batches:
    
    filename = [file.filename for file in files if file.id == batch.input_file_id]

    if filename and batch.output_file_id:

        batch_res = client.files.content(batch.output_file_id).text
        batch_res = [json.loads(line) for line in batch_res.splitlines()]
        all_batch_res.extend(batch_res)

if os.path.exists(out_path):
    with open(out_path, "r") as f:
        out_dict = json.load(f)
    if args.split not in out_dict.keys():
        out_dict[args.split] = {}
else:
    out_dict = {}
    
for i, dialog in enumerate(daily_dialog):
    for j, turn in enumerate(dialog):

        custom_id = f"{i}_{j}"
        res = [res["response"]["body"]["choices"][0]["message"]["content"] for res in all_batch_res if res["custom_id"] == custom_id]
        if res:
            out_dict[args.split][turn] = " ".join(res).strip()

with open(out_path, "w") as f:
    json.dump(out_dict, f)