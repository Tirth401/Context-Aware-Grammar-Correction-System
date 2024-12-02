# src/model_training.py

from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import os

def train_model():
    # Load the pre-trained T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Load the custom dataset
    dataset = load_from_disk('../data/custom_grammar_dataset')

    def preprocess_function(examples):
        inputs = ["grammar: " + text for text in examples['incorrect']]
        targets = examples['correct']
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir="../models/grammar_correction_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    # Save the fine-tuned model
    os.makedirs('../models/grammar_correction_model', exist_ok=True)
    model.save_pretrained("../models/grammar_correction_model")
    tokenizer.save_pretrained("../models/grammar_correction_model")
    print("Model training completed and saved.")

if __name__ == "__main__":
    train_model()