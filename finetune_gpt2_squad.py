# finetune_gpt2_squad.py
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "openai-community/gpt2"
OUTPUT_DIR = "artifacts/gpt2_squad_custom"

START_PHRASE = "That is a great question."
END_PHRASE = "Let me know if you have any other questions."

def build_text(example):
    """
    Turn each QA pair into a single training string that has your desired format.
    You can adjust this string format if your instructor wants something else.
    """
    q = example["question"].strip()
    a = example["answers"]["text"][0].strip() if len(example["answers"]["text"]) > 0 else ""
    text = (
        f"Q: {q}\n"
        f"A: {START_PHRASE} {a} {END_PHRASE}\n"
    )
    return {"text": text}

def main():
    # 1) Load SQuAD
    ds = load_dataset("rajpurkar/squad")

    # 2) Keep a small subset so it runs faster (you can change these sizes)
    train_small = ds["train"].select(range(2000))
    val_small = ds["validation"].select(range(500))

    train_small = train_small.map(build_text)
    val_small = val_small.map(build_text)

    # 3) Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT2 has no pad token by default, so we map pad -> eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    train_tok = train_small.map(tokenize_fn, batched=True, remove_columns=train_small.column_names)
    val_tok = val_small.map(tokenize_fn, batched=True, remove_columns=val_small.column_names)

    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 4) Training arguments
    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none",  # disable wandb/etc
)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )

    # 5) Train
    trainer.train()

    # 6) Save model + tokenizer to artifacts/
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
