from dataclasses import dataclass
import yaml


@dataclass
class AlbertLMConfig:
    model_name: str

    vocab_size: int

    hidden_size: int
    num_hidden_layers: int

    num_attention_heads: int
    num_key_value_heads: int

    intermediate_size: int

    max_position_embeddings: int

    activation: str
    normalization: str
    position_encoding: str

    attention_bias: bool
    mlp_bias: bool

    dropout: float

    tie_word_embeddings: bool

    dtype: str

    rope_theta: int
    rms_norm_eps: float


def load_config(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return AlbertLMConfig(**data)
