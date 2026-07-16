from __future__ import annotations

import contextlib
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import albertlm.model as model_module
from albertlm.linear import LinearMode
from albertlm.model import AlbertLM
from train import runtime_eval


class _Encoding:
    ids = [1, 2, 3, 4, 5, 6, 7]


class _Tokenizer:
    def encode(self, _prompt):
        return _Encoding()

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(value) for value in ids)


class _SampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), device="cuda"))
        self.calls = []

    def forward(self, input_ids, *, fp8_enabled=None):
        self.calls.append(
            {
                "length": int(input_ids.shape[1]),
                "fp8_enabled": fp8_enabled,
                "cuda_autocast": torch.is_autocast_enabled("cuda"),
            }
        )
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            32,
            device=input_ids.device,
        )
        return {"logits": logits}


class _Wrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _numpy_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


class PublicForwardTests(unittest.TestCase):
    def test_precision_override_is_keyword_only_and_preserves_default(self):
        config = SimpleNamespace(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
            attention_bias=False,
            mlp_bias=False,
            dropout=0.0,
            tie_word_embeddings=False,
            rope_theta=10000,
            rms_norm_eps=1.0e-6,
        )
        model = AlbertLM(config)
        model.linear_mode = LinearMode("te", True)
        observed = []

        def record(mode):
            observed.append(mode)
            return contextlib.nullcontext()

        tokens = torch.tensor([[1, 2, 3]])
        with mock.patch.object(model_module, "fp8_forward_context", record):
            model(tokens)
            model(tokens, fp8_enabled=False)

        self.assertTrue(observed[0].fp8_enabled)
        self.assertFalse(observed[1].fp8_enabled)
        self.assertTrue(model.fp8_enabled)
        with self.assertRaises(TypeError):
            model(tokens, None, False)

    def test_runtime_eval_does_not_reference_private_forward(self):
        source = Path(runtime_eval.__file__).read_text(encoding="utf-8")
        self.assertNotIn("_forward_impl", source)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RuntimeSampleTests(unittest.TestCase):
    def _run_case(self, wrapped):
        inner = _SampleModel()
        engine = _Wrapper(inner) if wrapped else inner
        engine.train()

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.random.get_rng_state().clone()
        cuda_state = [value.clone() for value in torch.cuda.get_rng_state_all()]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(runtime_eval, "_TOKENIZER", _Tokenizer()),
                mock.patch.object(
                    runtime_eval,
                    "_TOKENIZER_PATH",
                    root / "tokenizer.json",
                ),
                mock.patch.object(runtime_eval, "SAMPLE_DIR", root / "samples"),
                mock.patch.object(
                    runtime_eval,
                    "SAMPLES_JSONL_PATH",
                    root / "samples.jsonl",
                ),
                mock.patch.dict(
                    os.environ,
                    {
                        "SAMPLE_PROMPTS_JSON": '["short"]',
                        "SAMPLE_MAX_NEW_TOKENS": "16",
                        "SAMPLE_MAX_CONTEXT": "64",
                        "SAMPLE_TEMPERATURE": "0.8",
                        "SAMPLE_TOP_P": "0.9",
                    },
                    clear=False,
                ),
            ):
                result = runtime_eval.generate_samples(
                    engine,
                    optimizer_step=7984,
                    tokens_seen=1_046_478_848,
                )

        self.assertEqual([call["length"] for call in inner.calls], list(range(7, 23)))
        self.assertTrue(all(call["fp8_enabled"] is False for call in inner.calls))
        self.assertTrue(all(call["cuda_autocast"] for call in inner.calls))
        self.assertTrue(result["all_logits_finite"])
        self.assertEqual(result["samples"][0]["prompt_tokens"], 7)
        self.assertEqual(result["samples"][0]["generated_tokens"], 16)
        self.assertTrue(engine.training)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(_numpy_state_equal(np.random.get_state(), numpy_state))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_state))
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(torch.cuda.get_rng_state_all(), cuda_state)
            )
        )

    def test_bare_model_sample(self):
        self._run_case(wrapped=False)

    def test_common_module_wrapper_sample(self):
        self._run_case(wrapped=True)


if __name__ == "__main__":
    unittest.main()
