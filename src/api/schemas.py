"""Pydantic request and response schemas for the QA API."""

from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    context: str = Field(
        ...,
        min_length=10,
        description="The passage or document the question refers to",
        examples=["Paris is the capital and most populous city of France."],
    )
    question: str = Field(
        ...,
        min_length=5,
        description="The question to answer based on the context",
        examples=["What is the capital of France?"],
    )
    max_new_tokens: Optional[int] = Field(
        default=200,
        ge=10,
        le=512,
        description="Maximum number of new tokens to generate",
    )


class GenerateResponse(BaseModel):
    answer: str = Field(description="Generated answer text")
    context: str = Field(description="The input context")
    question: str = Field(description="The input question")
    model: str = Field(description="Model identifier (base + adapter)")
    tokens_generated: int = Field(description="Number of tokens in the generated answer")


class HealthResponse(BaseModel):
    status: str = Field(description="'healthy' when model is loaded, 'degraded' otherwise")
    model_loaded: bool
    model_name: str
    device: str = Field(description="Compute device: 'cuda' or 'cpu'")
    adapter_path: str
