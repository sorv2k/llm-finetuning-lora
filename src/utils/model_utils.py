"""
Shared utilities for loading models, tokenizers, and running inference.
Used by train.py, evaluate.py, and api/main.py.
"""

import logging
import torch
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

logger = logging.getLogger(__name__)

# QA prompt template used consistently across training, evaluation, and inference
PROMPT_TEMPLATE = """### Context:
{context}

### Question:
{question}

### Answer:
{answer}"""

ANSWER_DELIMITER = "### Answer:\n"


def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    """
    Load tokenizer and fix missing pad token for models like Phi-2.
    Sets padding_side='right' for causal LM training stability.
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    # Phi-2 and some other models lack a dedicated pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name: str,
    trust_remote_code: bool = True,
    load_in_4bit: bool = False,
    device_map: str = "auto",
):
    """
    Load a base causal LM from HuggingFace.

    Args:
        model_name: HuggingFace model ID (e.g. 'microsoft/phi-2')
        trust_remote_code: Required for Phi-2
        load_in_4bit: Enable QLoRA via bitsandbytes 4-bit quantisation
        device_map: 'auto' distributes across available hardware
    """
    logger.info(f"Loading base model: {model_name} (4bit={load_in_4bit})")

    kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
        "torch_dtype": torch.float16,
    }

    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def apply_lora(model, lora_cfg: dict):
    """
    Wrap a base model with LoRA adapters using PEFT.

    Args:
        model: Base PreTrainedModel
        lora_cfg: Dict from configs/lora_config.yaml under key 'lora'

    Returns:
        PeftModel with LoRA adapters applied
    """
    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg["target_modules"],
    )
    # Required for gradient checkpointing compatibility with PEFT
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_fine_tuned_model(
    base_model_name: str,
    adapter_path: str,
    device: Optional[str] = None,
):
    """
    Load a fine-tuned model by combining base weights with saved LoRA adapter.

    Returns:
        Tuple of (PeftModel in eval mode, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading fine-tuned model: {base_model_name} + adapter from {adapter_path}")

    tokenizer = load_tokenizer(base_model_name)
    base_model = load_base_model(
        base_model_name,
        device_map=device if device != "cpu" else None,
    )

    if device == "cpu":
        base_model = base_model.to(device)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    logger.info("Fine-tuned model loaded successfully")
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    context: str,
    question: str,
    generation_config: dict,
) -> str:
    """
    Format a QA prompt, generate tokens, and return only the answer portion.

    The model output includes the full prompt prefix; this function strips
    everything up to and including '### Answer:\\n' before returning.
    """
    prompt = PROMPT_TEMPLATE.format(context=context, question=question, answer="")
    # Trim the trailing empty answer so we only feed the prompt prefix
    prompt = prompt.rstrip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    input_length = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=generation_config.get("max_new_tokens", 200),
            temperature=generation_config.get("temperature", 0.7),
            do_sample=generation_config.get("do_sample", True),
            repetition_penalty=generation_config.get("repetition_penalty", 1.1),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][input_length:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer
