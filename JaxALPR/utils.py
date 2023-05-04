from .basic_types import KeyArray, Tuple
from .models import ModelConfig

import jax
import jax.numpy as jnp
import numpy as np
import random
from datetime import datetime
from ml_collections import ConfigDict
from typing import Tuple


DTYPE = {"float32": jnp.float32,
         "float64": jnp.float64,
         "float16": jnp.float16,
         "bfloat16": jnp.bfloat16}


def get_time() -> str:
  return datetime.strftime(datetime.now(), "%m%d-%H%M%S")


def set_random_seed(seed: int) -> KeyArray:
  random.seed(seed)
  np.random.seed(seed)

  return jax.random.PRNGKey(seed)


def parse_flags_to_model_config(config: ConfigDict) -> Tuple[ModelConfig, ModelConfig, ModelConfig]:
  train_config = ModelConfig(vocab_size=config.vocab_size,
                             logits_via_embedding=config.logits_via_embedding,
                             dtype=DTYPE[config.dtype],
                             embed_dim=config.embed_dim,
                             num_heads=config.n_heads,
                             num_layers=config.n_layers,
                             qkv_dim=config.qkv_dim,
                             mlp_dim=config.mlp_dim,
                             max_len=config.max_len,
                             dropout_rate=config.dropout_rate,
                             attention_dropout_rate=config.attention_dropout_rate,
                             deterministic=False,
                             decode=False)
  eval_config = ModelConfig(vocab_size=config.vocab_size,
                            logits_via_embedding=config.logits_via_embedding,
                            dtype=DTYPE[config.dtype],
                            embed_dim=config.embed_dim,
                            num_heads=config.n_heads,
                            num_layers=config.n_layers,
                            qkv_dim=config.qkv_dim,
                            mlp_dim=config.mlp_dim,
                            max_len=config.max_len,
                            dropout_rate=config.dropout_rate,
                            attention_dropout_rate=config.attention_dropout_rate,
                            deterministic=True,
                            decode=False)
  decode_config = ModelConfig(vocab_size=config.vocab_size,
                              logits_via_embedding=config.logits_via_embedding,
                              dtype=DTYPE[config.dtype],
                              embed_dim=config.embed_dim,
                              num_heads=config.n_heads,
                              num_layers=config.n_layers,
                              qkv_dim=config.qkv_dim,
                              mlp_dim=config.mlp_dim,
                              max_len=config.max_len,
                              dropout_rate=config.dropout_rate,
                              attention_dropout_rate=config.attention_dropout_rate,
                              deterministic=True,
                              decode=True)
  
  return train_config, eval_config, decode_config


def crop(image: np.ndarray, lp_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  _, width, _ = image.shape
  y_beg = max(lp_coords[1].min() - width // 2, 0)
  lp_coords[1] = lp_coords[1] - y_beg

  return image[y_beg:y_beg + width, :], lp_coords