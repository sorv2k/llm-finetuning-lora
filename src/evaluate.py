"""
Evaluate ROUGE-1 and ROUGE-L scores for baseline vs fine-tuned model on SQuAD.

Usage:
    python -m src.evaluate
    python -m src.evaluate --config configs/lora_config.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm

from src.utils.model_utils import (
    generate_answer,
    load_base_model,
    load_fine_tuned_model,
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


def evaluate_model(
    model,
    tokenizer,
    eval_samples: list,
    generation_config: dict,
    label: str = "model",
) -> dict:
    """
    Compute mean ROUGE-1 and ROUGE-L F-measures over eval_samples.

    Args:
        model: Loaded model (base or fine-tuned PeftModel)
        tokenizer: Corresponding tokenizer
        eval_samples: List of dicts with keys 'context', 'question', 'answer'
        generation_config: Dict of generation kwargs
        label: Display name for progress bar and results

    Returns:
        Dict with keys: label, rouge1, rougeL, num_samples
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_scores = []
    rougeL_scores = []

    model.eval()
    for sample in tqdm(eval_samples, desc=f"Evaluating {label}"):
        reference = sample["answer"]
        try:
            prediction = generate_answer(
                model=model,
                tokenizer=tokenizer,
                context=sample["context"],
                question=sample["question"],
                generation_config=generation_config,
            )
        except Exception as e:
            logger.warning(f"Generation failed for sample: {e}")
            prediction = ""

        scores = scorer.score(reference, prediction)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "label": label,
        "rouge1": round(sum(rouge1_scores) / len(rouge1_scores), 4) if rouge1_scores else 0.0,
        "rougeL": round(sum(rougeL_scores) / len(rougeL_scores), 4) if rougeL_scores else 0.0,
        "num_samples": len(rouge1_scores),
    }


def run_evaluation(cfg: dict) -> None:
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    inference_cfg = cfg["inference"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load eval samples from SQuAD validation set
    logger.info(f"Loading evaluation dataset: {training_cfg['dataset']}")
    raw = load_dataset(training_cfg["dataset"])
    eval_raw = raw["validation"].select(
        range(min(training_cfg["max_eval_samples"], len(raw["validation"])))
    )

    eval_samples = [
        {
            "context": row["context"],
            "question": row["question"],
            "answer": row["answers"]["text"][0] if row["answers"]["text"] else "",
        }
        for row in eval_raw
    ]

    generation_config = {
        "max_new_tokens": inference_cfg["max_new_tokens"],
        "temperature": inference_cfg["temperature"],
        "do_sample": inference_cfg["do_sample"],
        "repetition_penalty": inference_cfg["repetition_penalty"],
    }

    # --- Baseline: base model without any adapter ---
    logger.info("Evaluating baseline model (no adapter)...")
    base_tokenizer = load_tokenizer(model_cfg["name"], model_cfg["trust_remote_code"])
    base_model = load_base_model(
        model_cfg["name"],
        model_cfg["trust_remote_code"],
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        base_model = base_model.to(device)

    baseline_results = evaluate_model(
        model=base_model,
        tokenizer=base_tokenizer,
        eval_samples=eval_samples,
        generation_config=generation_config,
        label="Baseline (no LoRA)",
    )

    # Free baseline model memory before loading fine-tuned
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Fine-tuned: base model + LoRA adapter ---
    adapter_path = training_cfg["output_dir"]
    logger.info(f"Evaluating fine-tuned model from adapter: {adapter_path}")
    ft_model, ft_tokenizer = load_fine_tuned_model(
        base_model_name=model_cfg["name"],
        adapter_path=adapter_path,
        device=device,
    )

    finetuned_results = evaluate_model(
        model=ft_model,
        tokenizer=ft_tokenizer,
        eval_samples=eval_samples,
        generation_config=generation_config,
        label="Fine-tuned (LoRA)",
    )

    # --- Print comparison table ---
    print("\n" + "=" * 55)
    print(f"{'Model':<30} {'ROUGE-1':>10} {'ROUGE-L':>10}")
    print("-" * 55)
    print(f"{baseline_results['label']:<30} {baseline_results['rouge1']:>10.4f} {baseline_results['rougeL']:>10.4f}")
    print(f"{finetuned_results['label']:<30} {finetuned_results['rouge1']:>10.4f} {finetuned_results['rougeL']:>10.4f}")
    rouge1_delta = finetuned_results["rouge1"] - baseline_results["rouge1"]
    rougeL_delta = finetuned_results["rougeL"] - baseline_results["rougeL"]
    print("-" * 55)
    print(f"{'Delta':<30} {rouge1_delta:>+10.4f} {rougeL_delta:>+10.4f}")
    print("=" * 55)
    print(f"\nEvaluated on {baseline_results['num_samples']} SQuAD validation samples.")

    # --- Save results to JSON ---
    output = {
        "baseline": baseline_results,
        "fine_tuned": finetuned_results,
        "delta": {"rouge1": round(rouge1_delta, 4), "rougeL": round(rougeL_delta, 4)},
        "config": {
            "model": model_cfg["name"],
            "adapter_path": adapter_path,
            "num_samples": baseline_results["num_samples"],
            "lora_r": cfg["lora"]["r"],
            "lora_alpha": cfg["lora"]["lora_alpha"],
        },
    }
    out_path = Path("outputs/evaluation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_evaluation(cfg)
