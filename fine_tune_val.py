import json
import numpy as np
import pandas as pd
import torch
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

HUGGING_FACE_TOKEN = "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" 
word_separator_token = "@"

class CompoundSegmentationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    original, segmented = line.strip().split('\t')
                    data.append((original, segmented))
        return data

    def generate_possibilities(self, original, segmented):
        possibilities = []
        segment_boundaries = set([i for i, char in enumerate(segmented) if char == '@'])
        for i in range(1, len(original)):
            possibility = f"{original[:i]}@{original[i:]}"
            label = int(i in segment_boundaries)
            possibilities.append((possibility, label))
        return possibilities

    def __len__(self):
        return sum(len(self.generate_possibilities(orig, seg)) for orig, seg in self.data)

    def __getitem__(self, idx):
        for original, segmented in self.data:
            possibilities = self.generate_possibilities(original, segmented)
            if idx < len(possibilities):
                possibility, label = possibilities[idx]
                text = f"{original} {word_separator_token} {possibility}"
                encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length)
                return {
                    'input_ids': torch.tensor(encoding['input_ids']),
                    'attention_mask': torch.tensor(encoding['attention_mask']),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
            idx -= len(possibilities)
        raise IndexError("Index out of range")

    def get_class_weights(self):
        all_labels = [label for _, seg in self.data for _, label in self.generate_possibilities(_, seg)]
        return compute_class_weight('balanced', classes=np.array([0, 1]), y=all_labels)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if logits.size(0) != labels.size(0):
            raise ValueError(f"Logits batch size ({logits.size(0)}) does not match labels batch size ({labels.size(0)})")

        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred, clf_metrics):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return clf_metrics.compute(predictions=predictions, references=labels)


def main(lang):
    lang = "afrikaans"
    model_id = "./afro-xlmr-mini"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGING_FACE_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, token=HUGGING_FACE_TOKEN)

    tokenizer.add_special_tokens({"additional_special_tokens": ["@"]})
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = CompoundSegmentationDataset(f"./Dataset/{lang}_train.txt", tokenizer)
    val_dataset = CompoundSegmentationDataset(f"./Dataset/{lang}_val.txt", tokenizer)

    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    training_args = TrainingArguments(
        output_dir=f"/home/jdiegaardt/lustre/NLP_afri/out/trained-model-{lang}-def",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=30,
        logging_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        warmup_steps=20,
        gradient_accumulation_steps=16,   # Add gradient accumulation
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, clf_metrics),  # Pass clf_metrics here,
    )

    class_weights = train_dataset.get_class_weights()
    trainer.loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights.astype(np.float32)).to(model.device)
    )
     # Synchronize before starting the training
    torch.cuda.synchronize()  # Ensure no pending GPU operations

    trainer.train()

    # Synchronize after training
    torch.cuda.synchronize()  # Ensure all operations are completed

    # Save the fine-tuned model and tokenizer after training
    model.save_pretrained(f"/home/jdiegaardt/lustre/NLP_afri/out/fine-tuned-model-{lang}-def")
    tokenizer.save_pretrained(f"/home/jdiegaardt/lustre/NLP_afri/out/fine-tuned-tokenizer-{lang}-def")

    print(f"Fine-tuned model and tokenizer saved in /out/fine-tuned-model-{lang}-def and /out/fine-tuned-tokenizer-{lang}-def")

if __name__ == "__main__":
    for lang in ("afrikaans",):
        main(lang)
