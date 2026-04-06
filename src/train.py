"""
Fine-tune a base LLM with LoRA/PEFT on the SQuAD QA dataset.

Usage:
    python -m src.train
    python -m src.train --config configs/lora_config.yaml
"""

import argparse
import logging
import os
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.utils.model_utils import (
    ANSWER_DELIMITER,
    PROMPT_TEMPLATE,
    apply_lora,
    load_base_model,
    load_tokenizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/lora_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess_function(examples, tokenizer, max_seq_length: int):
    """
    Tokenise SQuAD examples into (input_ids, attention_mask, labels).

    Labels are masked with -100 for context+question tokens so the model
    only learns to predict the answer tokens.
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    # Tokenise the answer-delimiter prefix to find where the answer starts
    delimiter_ids = tokenizer.encode(ANSWER_DELIMITER, add_special_tokens=False)
    delimiter_len = len(delimiter_ids)

    for i in range(len(examples["context"])):
        context = examples["context"][i]
        question = examples["question"][i]
        # SQuAD answers field is a dict with 'text' list; take the first answer
        answer = examples["answers"][i]["text"][0] if examples["answers"][i]["text"] else ""

        full_prompt = PROMPT_TEMPLATE.format(
            context=context, question=question, answer=answer
        )

        tokenised = tokenizer(
            full_prompt,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

        input_ids = tokenised["input_ids"]
        attention_mask = tokenised["attention_mask"]
        labels = list(input_ids)  # copy

        # Find the last occurrence of the delimiter token sequence in input_ids
        # and mask everything before (and including) it with -100
        answer_start_idx = None
        for j in range(len(input_ids) - delimiter_len + 1):
            if input_ids[j : j + delimiter_len] == delimiter_ids:
                answer_start_idx = j + delimiter_len
                # Use the last match in case delimiter appears in context

        if answer_start_idx is not None:
            for j in range(answer_start_idx):
                labels[j] = -100
        else:
            # Fallback: mask everything (safe default, model learns nothing harmful)
            labels = [-100] * len(labels)

        # Also mask padding tokens
        for j in range(len(labels)):
            if input_ids[j] == tokenizer.pad_token_id:
                labels[j] = -100

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def prepare_dataset(cfg: dict, tokenizer):
    """
    Load SQuAD and preprocess into tokenised train/eval datasets.
    """
    training_cfg = cfg["training"]
    logger.info(f"Loading dataset: {training_cfg['dataset']}")

    raw = load_dataset(training_cfg["dataset"])
    train_raw = raw["train"].select(range(min(training_cfg["max_train_samples"], len(raw["train"]))))
    eval_raw = raw["validation"].select(range(min(training_cfg["max_eval_samples"], len(raw["validation"]))))

    logger.info(f"Train samples: {len(train_raw)}, Eval samples: {len(eval_raw)}")

    cols_to_remove = train_raw.column_names

    train_ds = train_raw.map(
        lambda examples: preprocess_function(examples, tokenizer, training_cfg["max_seq_length"]),
        batched=True,
        remove_columns=cols_to_remove,
        desc="Preprocessing train set",
    )
    eval_ds = eval_raw.map(
        lambda examples: preprocess_function(examples, tokenizer, training_cfg["max_seq_length"]),
        batched=True,
        remove_columns=eval_raw.column_names,
        desc="Preprocessing eval set",
    )

    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    return train_ds, eval_ds


def build_training_args(cfg: dict) -> TrainingArguments:
    training_cfg = cfg["training"]
    return TrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        warmup_ratio=training_cfg["warmup_ratio"],
        lr_scheduler_type=training_cfg["lr_scheduler"],
        fp16=training_cfg["fp16"],
        logging_steps=training_cfg["logging_steps"],
        save_steps=training_cfg["save_steps"],
        evaluation_strategy="steps",
        eval_steps=training_cfg["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        seed=training_cfg["seed"],
        dataloader_pin_memory=False,
    )


def train(cfg: dict) -> None:
    set_seed(cfg["training"]["seed"])

    model_cfg = cfg["model"]
    tokenizer = load_tokenizer(model_cfg["name"], model_cfg["trust_remote_code"])
    base_model = load_base_model(model_cfg["name"], model_cfg["trust_remote_code"])
    model = apply_lora(base_model, cfg["lora"])

    train_ds, eval_ds = prepare_dataset(cfg, tokenizer)

    training_args = build_training_args(cfg)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    output_dir = cfg["training"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Adapter weights saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
