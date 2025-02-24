from datasets import load_dataset
from transformers import RobertaTokenizer, Trainer, RobertaForSequenceClassification, TrainingArguments

from helpers import get_emotion_labels

parser = argparse.ArgumentParser()
parser.add_argument("-lg", "--language", default="nl", type=str)
args = parser.parse_args()

translator_model_name = "GPT-4o-mini"
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
                prepped_dataset[split].append({
                    "text": stored_translations[split][text]
                    "label": labels[i][j]
                })

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

print(dataset['train'][0])
# Output: {'text': 'Dit is een voorbeeldzin.', 'label': 1}

tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

tokenized_datasets.set_format("torch")
model = RobertaForSequenceClassification.from_pretrained("pdelobelle/robbert-v2-dutch-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="files",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)