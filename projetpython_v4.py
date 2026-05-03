import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import evaluate
import numpy as np

# fonction d'évaluation (les métriques)
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    }

# charger le fichier
dataset = load_dataset("csv", data_files="parlacap.tsv", sep="\t", split="train")
dataset = dataset.filter(lambda x: x["speaker_party"] is not None and x["speaker_party"] != "-")

# nettoyage des données
min_words = 15
dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"].split()) > min_words)

# Regroupement des partis avec une fréquence < 5%
threshold = 0.05

counts = dataset.to_pandas()["speaker_party"].value_counts(normalize=True)
petits_partis = counts[counts < threshold].index.tolist()

def group_small_parties(example):
    if example["speaker_party"] in petits_partis:
        example["speaker_party"] = "AUTRE"
    return example

dataset = dataset.map(group_small_parties)

# labéliser les données
dataset = dataset.class_encode_column("speaker_party")
label_names = dataset.features["speaker_party"].names
dataset = dataset.rename_column("speaker_party", "label")

# On réserve 20% des données pour le test avec un split stratifié
dataset_split = dataset.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="label"
)

# choix du modèle : modèle plus fort que distilcamembert
model_name = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenisation avec plus de contexte
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_data = dataset_split.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Calcul des poids des classes pour mieux gérer le déséquilibre
train_labels = np.array(tokenized_data["train"]["label"])
class_counts = np.bincount(train_labels, minlength=len(label_names))

class_weights = class_counts.sum() / (len(label_names) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float)

print("Poids des classes :")
for label, weight in zip(label_names, class_weights):
    print(label, ":", float(weight))


# Dictionnaire/table de correspondance
id2label = {i: label_names[i] for i in range(len(label_names))}
label2id = {label_names[i]: i for i in range(len(label_names))}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id
)

# entraînement du modèle
training_args = TrainingArguments(
    output_dir="output_v4",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    push_to_hub=False,
)

# Trainer personnalisé avec pondération des classes
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels") if "labels" in inputs else inputs.pop("label")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# entraînement
trainer.train()

# évaluation finale
results = trainer.evaluate()
print("Résultats finaux :")
print(results)
