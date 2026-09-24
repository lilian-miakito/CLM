"""Local MPS path tests with a tiny random Qwen3 model; no model download."""
import base64
import os
import tempfile
import unittest

import numpy as np
import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from clm.heads import default_device, make_head
from clm.engine import Engine
from clm.embedder import Embedder
from clm.mps_embed import MpsEncoder, create_app


class MacosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        config = Qwen3Config(vocab_size=16, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16,
                             max_position_embeddings=128)
        torch.manual_seed(7)
        # The published Qwen3-8B checkpoint is packaged as a causal LM.
        Qwen3ForCausalLM(config).save_pretrained(cls.tmp.name)
        tokenizer = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
                                                "hello": 3, "world": 4, "long": 5}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]",
                                eos_token="[EOS]", unk_token="[UNK]").save_pretrained(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_embedding_contract_and_last_non_padding_token(self):
        encoder = MpsEncoder(self.tmp.name, "cpu", max_tokens=16, batch_size=2)
        vectors, tokens = encoder.embed(["hello", "hello world"])
        self.assertEqual(vectors.shape, (2, 64))
        self.assertEqual(tokens, 3)
        padded = encoder.tokenizer(["hello", "hello world"], padding=True, return_tensors="pt")
        with torch.inference_mode():
            hidden = encoder.model(**padded, use_cache=False).last_hidden_state
        np.testing.assert_allclose(vectors[0], hidden[0, 0].numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(vectors[1], hidden[1, 1].numpy(), rtol=1e-5, atol=1e-5)

        client = TestClient(create_app(encoder))
        self.assertEqual(client.get("/v1/models").status_code, 200)
        r = client.post("/v1/embeddings", json={"model": "qwen3-8b", "input": ["hello", "hello world"],
                                                "encoding_format": "base64"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["usage"]["prompt_tokens"], 3)
        decoded = np.frombuffer(base64.b64decode(body["data"][0]["embedding"]), dtype=np.float32)
        np.testing.assert_allclose(decoded, vectors[0], rtol=1e-5, atol=1e-5)
        embedder = Embedder(url="http://testserver/v1/embeddings", model="qwen3-8b")
        embedder.session = client
        normalised, spent = embedder.embed(["hello", "hello world"])
        self.assertEqual(spent, 3)
        np.testing.assert_allclose(normalised[0], vectors[0] / np.linalg.norm(vectors[0]),
                                   rtol=1e-5, atol=1e-5)
        self.assertEqual(client.post("/v1/embeddings", json={"model": "qwen3-8b", "input": []}).status_code, 422)
        self.assertEqual(client.post("/v1/embeddings", json={"model": "wrong", "input": "hello"}).status_code, 422)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_embedding_and_projection(self):
        from clm.cache import VectorArena

        self.assertEqual(default_device(), "mps")
        encoder = MpsEncoder(self.tmp.name, "mps", max_tokens=16)
        vectors, tokens = encoder.embed(["hello world"])
        self.assertEqual(vectors.shape, (1, 64))
        self.assertEqual(tokens, 2)
        head = make_head(width=32, depth=2, proj=16, hidden=64).to("mps")
        with torch.inference_mode():
            projected = head(torch.from_numpy(vectors).to("mps"))
        self.assertEqual(tuple(projected.shape), (1, 16))
        self.assertEqual(projected.device.type, "mps")
        arena = VectorArena("mps", "1MiB")
        self.assertGreater(arena.reserved_mb, 0)

        checkpoint = os.path.join(self.tmp.name, "tiny-head.pt")
        torch.save({"cfg": {"width": 32, "depth": 2, "hidden_size": 64, "projection_dim": 16},
                    "state_head": make_head(32, 2, 16, hidden=64).state_dict(),
                    "action_head": make_head(32, 2, 16, hidden=64).state_dict(),
                    "logit_scale": torch.tensor(0.0)}, checkpoint)
        engine = Engine(embedder=encoder, checkpoint=checkpoint, device="mps")
        self.assertIsNone(engine.arena)  # cache stays off by default on unified memory
        answer = engine.answer("hello", {"team": {"type": "choice", "instructions": "world",
                                                  "criteria": {"a": "hello", "b": "world"}}})
        self.assertIn(answer["answers"]["team"]["choice"], ("a", "b"))
        self.assertEqual(answer["usage"]["billing_units"], 1)


if __name__ == "__main__":
    unittest.main()
