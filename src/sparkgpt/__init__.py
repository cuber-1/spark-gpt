"""SparkGPT: a compact GPT training stack for NVIDIA DGX Spark."""

from sparkgpt.config import ModelConfig, TrainConfig, load_config
from sparkgpt.model import GPT

__all__ = ["GPT", "ModelConfig", "TrainConfig", "load_config"]
__version__ = "0.1.0"
