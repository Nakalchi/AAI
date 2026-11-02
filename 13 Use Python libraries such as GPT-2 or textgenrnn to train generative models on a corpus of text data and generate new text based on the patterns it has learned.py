python -m pip install transformers datasets torch

import os
import urllib.request
import torch

from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)


# Step 1: Set Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Step 2: Download Tiny Shakespeare Dataset
print("Loading Tiny Shakespeare dataset...")

data_url = (
    "https://raw.githubusercontent.com/"
    "karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

data_file = "input.txt"

if not os.path.exists(data_file):
    urllib.request.urlretrieve(data_url, data_file)

print("Dataset downloaded successfully!")

with open(data_file, "r", encoding="utf-8") as f:
    text = f.read()

print("Total characters:", len(text))


# Step 3: Split Dataset
split_index = int(len(text) * 0.9)

train_text = text[:split_index]
test_text = text[split_index:]


# Step 4: Load GPT-2 Model and Tokenizer
print("Loading GPT-2 tokenizer and model...")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id


# Step 5: Tokenize Dataset in Chunks
def prepare_dataset(text):

    block_size = 512

    tokens = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False
    )["input_ids"]

    blocks = []

    for i in range(0, len(tokens) - block_size + 1, block_size):
        blocks.append(tokens[i:i + block_size])

    return Dataset.from_dict({
        "input_ids": blocks
    })


print("Tokenizing training data...")

tokenized_train = prepare_dataset(train_text)

print("Tokenizing testing data...")

tokenized_test = prepare_dataset(test_text)

print("Training blocks:", len(tokenized_train))
print("Testing blocks:", len(tokenized_test))


# Step 6: Define Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# Step 7: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    num_train_epochs=3,

    learning_rate=5e-5,

    weight_decay=0.01,

    save_strategy="epoch",

    save_total_limit=2,

    logging_steps=10,

    report_to="none",

    dataloader_num_workers=0
)


# Step 8: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator
)


# Step 9: Train GPT-2 Model
print("Starting GPT-2 training...")

trainer.train()


# Step 10: Save Fine-Tuned Model
model.save_pretrained("./fine_tuned_gpt2")

tokenizer.save_pretrained("./fine_tuned_gpt2")

print("Model saved successfully!")


# Step 11: Generate New Text
print("Generating new text...")

model.to(device)
model.eval()

input_text = "To be or not to be, that is the"

inputs = tokenizer(
    input_text,
    return_tensors="pt"
).to(device)

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )


# Step 12: Display Generated Text
generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nGenerated Text:\n")
print(generated_text)

print("\nProgram completed successfully!")
