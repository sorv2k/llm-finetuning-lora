"""
FastAPI application serving a LoRA fine-tuned QA model.

Endpoints:
    GET  /health    — Model and service status
    POST /generate  — Answer a question given a context passage

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Environment variables:
    MODEL_NAME      HuggingFace model ID (default: microsoft/phi-2)
    ADAPTER_PATH    Path to saved LoRA adapter weights (default: ./outputs/adapter_weights)
    TEMPERATURE     Sampling temperature (default: 0.7)
    LOG_LEVEL       Logging level (default: INFO)
"""

import logging
import os
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.api.schemas import GenerateRequest, GenerateResponse, HealthResponse
from src.utils.model_utils import generate_answer, load_fine_tuned_model

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Mutable module-level state — populated during lifespan startup
model_state: dict = {
    "model": None,
    "tokenizer": None,
    "loaded": False,
    "model_name": os.getenv("MODEL_NAME", "microsoft/phi-2"),
    "adapter_path": os.getenv("ADAPTER_PATH", "./outputs/adapter_weights"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; release GPU memory on shutdown."""
    logger.info(
        f"Loading model {model_state['model_name']} "
        f"with adapter {model_state['adapter_path']} "
        f"on {model_state['device']}"
    )
    try:
        model_state["model"], model_state["tokenizer"] = load_fine_tuned_model(
            base_model_name=model_state["model_name"],
            adapter_path=model_state["adapter_path"],
            device=model_state["device"],
        )
        model_state["loaded"] = True
        logger.info("Model loaded successfully — API is ready")
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        # Allow the server to start in degraded mode so /health can report the issue

    yield

    # Cleanup
    logger.info("Shutting down — releasing model from memory")
    model_state["model"] = None
    model_state["tokenizer"] = None
    model_state["loaded"] = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="LLM Fine-Tuning LoRA — QA API",
    description="Answer questions about a provided context using a LoRA fine-tuned Phi-2 model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check():
    """
    Returns model load status and runtime metadata.
    status='healthy' means a model is loaded and ready to serve requests.
    """
    return HealthResponse(
        status="healthy" if model_state["loaded"] else "degraded",
        model_loaded=model_state["loaded"],
        model_name=model_state["model_name"],
        device=model_state["device"],
        adapter_path=model_state["adapter_path"],
    )


@app.post("/generate", response_model=GenerateResponse, summary="Generate an answer")
async def generate(request: GenerateRequest):
    """
    Given a context passage and a question, return a generated answer.

    The model answers exclusively from the provided context (extractive-style QA).
    Increase max_new_tokens for longer or more detailed answers.
    """
    if not model_state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check /health for details.",
        )

    generation_config = {
        "max_new_tokens": request.max_new_tokens,
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
        "do_sample": True,
        "repetition_penalty": 1.1,
    }

    try:
        answer = generate_answer(
            model=model_state["model"],
            tokenizer=model_state["tokenizer"],
            context=request.context,
            question=request.question,
            generation_config=generation_config,
        )
    except Exception as exc:
        logger.error(f"Generation error: {exc}")
        raise HTTPException(status_code=500, detail="Generation failed. See server logs.")

    tokens_generated = len(model_state["tokenizer"].encode(answer))

    return GenerateResponse(
        answer=answer,
        context=request.context,
        question=request.question,
        model=f"{model_state['model_name']} + LoRA",
        tokens_generated=tokens_generated,
    )
