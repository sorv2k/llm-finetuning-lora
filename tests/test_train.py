"""
Unit tests for training utilities — label masking, config loading.
No model weights are downloaded; tests use a lightweight tokenizer mock.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestLoadConfig:
    def test_load_config_returns_dict(self, tmp_path):
        from src.train import load_config

        cfg_content = """
model:
  name: "microsoft/phi-2"
  trust_remote_code: true
lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules: ["q_proj"]
training:
  dataset: "rajpurkar/squad"
  max_train_samples: 100
  max_eval_samples: 10
  max_seq_length: 128
  epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 1
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler: "cosine"
  fp16: false
  logging_steps: 10
  save_steps: 50
  output_dir: "/tmp/test_adapter"
  seed: 42
inference:
  max_new_tokens: 50
  temperature: 0.7
  do_sample: true
  repetition_penalty: 1.1
"""
        cfg_file = tmp_path / "lora_config.yaml"
        cfg_file.write_text(cfg_content)

        cfg = load_config(str(cfg_file))
        assert isinstance(cfg, dict)
        assert cfg["model"]["name"] == "microsoft/phi-2"
        assert cfg["lora"]["r"] == 8
        assert cfg["training"]["epochs"] == 1


class TestPreprocessFunction:
    """Test that label masking correctly masks context+question tokens."""

    def _make_mock_tokenizer(self, delimiter_ids: list[int]):
        """
        Build a minimal tokenizer mock.
        encode() returns delimiter_ids when called with ANSWER_DELIMITER,
        otherwise returns a predictable sequence.
        """
        from src.utils.model_utils import ANSWER_DELIMITER

        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok.eos_token_id = 2

        def encode_side_effect(text, add_special_tokens=True):
            if text == ANSWER_DELIMITER:
                return delimiter_ids
            return list(range(1, 21))  # 20 fake token IDs

        mock_tok.encode.side_effect = encode_side_effect

        # __call__ mimics tokenizer(text, ...) → dict with input_ids, attention_mask
        def call_side_effect(text, truncation=False, max_length=512, padding=None):
            ids = list(range(1, max_length + 1))
            # Embed delimiter at position 10
            for i, d in enumerate(delimiter_ids):
                ids[10 + i] = d
            ids = ids[:max_length]
            mask = [1] * max_length
            return {"input_ids": ids, "attention_mask": mask}

        mock_tok.side_effect = call_side_effect
        return mock_tok

    def test_labels_masked_before_answer_delimiter(self):
        from src.train import preprocess_function

        delimiter_ids = [100, 101, 102]
        tokenizer = self._make_mock_tokenizer(delimiter_ids)

        examples = {
            "context": ["Paris is the capital of France."],
            "question": ["What is the capital?"],
            "answers": [{"text": ["Paris"], "answer_start": [0]}],
        }

        result = preprocess_function(examples, tokenizer, max_seq_length=50)

        labels = result["labels"][0]
        # All positions before delimiter+len should be -100
        answer_start = 10 + len(delimiter_ids)
        for i in range(answer_start):
            assert labels[i] == -100, f"Position {i} should be masked but got {labels[i]}"

    def test_output_keys_present(self):
        from src.train import preprocess_function

        tokenizer = self._make_mock_tokenizer([100, 101])

        examples = {
            "context": ["Some context."],
            "question": ["A question?"],
            "answers": [{"text": ["answer"], "answer_start": [0]}],
        }

        result = preprocess_function(examples, tokenizer, max_seq_length=50)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_empty_answer_falls_back_to_full_mask(self):
        """When delimiter tokens never appear in input_ids, all labels should be -100."""
        from src.train import preprocess_function

        # Use a delimiter that will NOT be embedded by the mock (multi-token, unusual IDs)
        delimiter_ids = [999]

        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok.eos_token_id = 2

        from src.utils.model_utils import ANSWER_DELIMITER

        def encode_side_effect(text, add_special_tokens=True):
            if text == ANSWER_DELIMITER:
                return delimiter_ids
            return list(range(1, 21))

        mock_tok.encode.side_effect = encode_side_effect

        # __call__: return ids that do NOT contain 999 — delimiter is absent
        def call_side_effect(text, truncation=False, max_length=512, padding=None):
            ids = [1] * max_length   # all 1s — 999 never appears
            mask = [1] * max_length
            return {"input_ids": ids, "attention_mask": mask}

        mock_tok.side_effect = call_side_effect

        examples = {
            "context": ["Context."],
            "question": ["Question?"],
            "answers": [{"text": [], "answer_start": []}],
        }

        result = preprocess_function(examples, mock_tok, max_seq_length=50)
        labels = result["labels"][0]
        # Delimiter not found → fallback: all labels should be -100
        assert all(l == -100 for l in labels)
