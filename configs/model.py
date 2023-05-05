from ml_collections import ConfigDict


def get_config() -> ConfigDict:
  config = ConfigDict()

  config.vocab_size = 68
  config.logits_via_embedding = True
  config.dtype = "bfloat16"
  config.embed_dim = 256
  config.n_heads = 8
  config.n_layers = 6
  config.qkv_dim = 512
  config.mlp_dim = 2048
  config.max_len = 256
  config.dropout_rate = 0.1
  config.attention_dropout_rate = 0.1

  return config