from .basic_types import KeyArray, Tuple, List
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


class Vocabulary(object):
  def __init__(self, max_len: int):
    self._keys = ["I", "O", "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "澳"] + ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    self._vocab = {k: i for i, k in enumerate(self._keys)}
    self._max_len = max_len

  def decode(self, input_ids: List[int]) -> List[str]:
    return [self._keys[input_id] for input_id in input_ids]
  
  def batch_decode(self, input_ids: List[List[int]]) -> List[List[str]]:
    return [self.decode(inp) for inp in input_ids]
  
  def encode(self, chars: str) -> List[int]:
    return [self._vocab[c] for c in chars]
  
  def batch_encode(self, b_chars: List[str]) -> List[List[int]]:
    res = [self.encode(chars) for chars in b_chars]
    return list(map(lambda x: [0] + x + [1] * (self._max_len - 1 - len(x)), res))
  
  @property
  def keys(self):
    return self._keys


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