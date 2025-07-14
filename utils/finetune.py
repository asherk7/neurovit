import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

dataset = load_dataset("bigbio/med_qa")

def format_example(example):
    question = example["question"]
    choices = example["choices"]
    answer = example["answer"]
    choice_str = "\n".join(choices)
    text = f"Question:\n{question}\n\nChoices:\n{choice_str}\n\nAnswer:\n{answer}"
    return {"text": text}

dataset = dataset.map(format_example)
dataset = dataset["train"] 
print("test")

model_name = "google/gemma-2b-it"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("test")

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = dataset.map(tokenize, batched=True)
print("test")

training_args = TrainingArguments(
    output_dir="./finetuned_gemma_medqa",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=2,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="none"
)
print("test")

def data_collator(batch):
    return {
        "input_ids": torch.stack([torch.tensor(example["input_ids"]) for example in batch]),
        "attention_mask": torch.stack([torch.tensor(example["attention_mask"]) for example in batch]),
        "labels": torch.stack([torch.tensor(example["input_ids"]) for example in batch]),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)
print("final")

trainer.train()

model.save_pretrained("./finetuned_gemma_medqa_lora")
tokenizer.save_pretrained("./finetuned_gemma_medqa_lora")
print("Finetuning complete")
