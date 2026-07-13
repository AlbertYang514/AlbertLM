from albertlm.config import load_config
import sys


def estimate(config):

    vocab = config.vocab_size
    h = config.hidden_size
    layers = config.num_hidden_layers

    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads

    head_dim = h // heads

    # Embedding
    embedding = vocab * h

    # Attention
    # Q: hidden -> hidden
    q_proj = h * h

    # K/V: hidden -> kv_heads * head_dim
    kv_dim = kv_heads * head_dim

    k_proj = h * kv_dim
    v_proj = h * kv_dim

    # output projection
    o_proj = h * h

    attention_layer = (
        q_proj +
        k_proj +
        v_proj +
        o_proj
    )

    attention = layers * attention_layer


    # SwiGLU
    # gate + up + down

    ffn_layer = (
        h * config.intermediate_size * 2
        +
        config.intermediate_size * h
    )

    ffn = layers * ffn_layer


    lm_head = vocab * h


    total = (
        embedding
        +
        attention
        +
        ffn
        +
        lm_head
    )


    print("=" * 40)
    print(config.model_name)
    print("=" * 40)

    print(f"Embedding: {embedding/1e6:.2f} M")
    print(f"Attention: {attention/1e6:.2f} M")
    print(f"FFN:       {ffn/1e6:.2f} M")
    print(f"LM Head:   {lm_head/1e6:.2f} M")
    print("-" * 40)
    print(f"TOTAL:     {total/1e6:.2f} M")


if __name__ == "__main__":
    cfg = load_config(sys.argv[1])
    estimate(cfg)
