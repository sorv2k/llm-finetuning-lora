# LLM Fine-Tuning with LoRA

Fine-tunes the Phi-2 language model using LoRA (Low-Rank Adaptation) on the SQuAD question-answering dataset and serves it via a REST API. LoRA allows efficient fine-tuning by training only small adapter weights instead of the full model, reducing both training time and storage requirements.

## Tech Stack

Python 3.10, PyTorch, Transformers, PEFT, SQuAD dataset, FastAPI, Docker

## Prerequisites

- Python 3.10 or higher
- CUDA 11.8 or higher (optional, CPU works but is slower)
- 6 GB VRAM for training, 4 GB for inference
- Docker for containerized deployment

## Running Locally

### Training

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
python -m src.train
```

Adapter weights are saved to `outputs/adapter_weights/`. Training takes approximately 2 hours on A100 or 8 hours on T4.

### Evaluation

```bash
python -m src.evaluate
```

This compares ROUGE-1 and ROUGE-L scores between the base model and fine-tuned adapter on 200 SQuAD validation samples.

### API Server

```bash
uvicorn src.api.main:app --reload --port 8000
```

Or with Docker:

```bash
docker build -t lora-qa .
docker run -p 8000:8000 -v ./outputs:/app/outputs lora-qa
```

API documentation is available at http://localhost:8000/docs. Send questions with POST requests to `/generate` endpoint including context and question fields.
