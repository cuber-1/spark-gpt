import unittest

from sparkgpt.config import ModelConfig, load_config
from sparkgpt.model import GPT


class ConfigTests(unittest.TestCase):
    def test_parameter_accounting_matches_model(self) -> None:
        config = ModelConfig(
            vocab_size=256,
            context_length=32,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=64,
            intermediate_size=176,
        )
        self.assertEqual(config.parameter_count(), GPT(config).parameter_count())

    def test_published_parameter_targets(self) -> None:
        self.assertEqual(
            load_config("configs/spark_300m.toml").model.parameter_count(), 304_137_216
        )
        self.assertEqual(
            load_config("configs/spark_8b_sanity.toml").model.parameter_count(),
            7_986_253_824,
        )

    def test_invalid_head_geometry_fails(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(
                vocab_size=256,
                context_length=32,
                n_layer=2,
                n_head=3,
                n_kv_head=1,
                n_embd=64,
                intermediate_size=176,
            )

    def test_invalid_optimizer_ranges_fail(self) -> None:
        model = ModelConfig(
            vocab_size=256,
            context_length=32,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=64,
            intermediate_size=176,
        )
        from sparkgpt.config import TrainConfig

        with self.assertRaises(ValueError):
            TrainConfig(model=model, min_lr=-1e-4)
        with self.assertRaises(ValueError):
            TrainConfig(model=model, grad_clip=-1.0)


if __name__ == "__main__":
    unittest.main()
