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
files = sorted([f for f in os.listdir("files") if f.startswith(file_name)], key=lambda x: int(x.split('_')[-1].split('.')[0]))

batches = client.batches.list()
files = [f for f in client.files.list() if f.filename in files]
daily_dialog = load_dataset("li2017dailydialog/daily_dialog")[args.split]["dialog"]

out_path = os.path.join("files", f"translations_{model_name}_{args.language}.json")
all_batch_res = []
offset = 0

sorted_files = sorted(files, key=lambda f: int(f.filename.split('_')[-1].split('.')[0]))

for file in sorted_files:
    batch = next((b for b in batches if b.input_file_id == file.id), None)
    file_index = int(file.filename.split('_')[-1].split('.')[0])
    
    if batch and batch.output_file_id:
        batch_res = client.files.content(batch.output_file_id).text
        batch_res = [json.loads(line) for line in batch_res.splitlines()]
        
        if file_index == 1:
            all_batch_res.extend(batch_res)
            offset = max(int(res["custom_id"].split("_")[0]) for res in batch_res) + 1
        else:
            adjusted_batch_res = []
            # Check if we need to apply offset by looking at first element
            first_i = int(batch_res[0]["custom_id"].split("_")[0])
            needs_offset = first_i != offset
            
            for res in batch_res:
                i, j = map(int, res["custom_id"].split("_"))
                new_res = res.copy()
                new_res["custom_id"] = f"{i + offset if needs_offset else i}_{j}"
                adjusted_batch_res.append(new_res)
            all_batch_res.extend(adjusted_batch_res)
            offset += len(batch_res) if needs_offset else 0

if os.path.exists(out_path):
    with open(out_path, "r") as f:
        out_dict = json.load(f)
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