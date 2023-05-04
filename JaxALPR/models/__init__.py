import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax import struct
from functools import partial
from typing import Any, Callable, Optional


ZERO = 1e-6


@struct.dataclass
class ModelConfig(object):
  vocab_size: int = 68
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  embed_dim: int = 256
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 256
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Any = nn.initializers.xavier_normal()
  bias_init: Any = nn.initializers.zeros_init()


class ResBlock(nn.Module):
  config: ModelConfig
  n_features: int

  @nn.compact
  def __call__(self, inp: jax.Array, train: bool = True) -> jax.Array:
    config = self.config
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=config.dtype)
    conv = partial(nn.Conv,
                   dtype=config.dtype,
                   kernel_init=config.kernel_init,
                   use_bias=False)
    
    x = conv(self.n_features,
             kernel_size=(3, 3),
             strides=1,
             padding="SAME")(inp)
    x = nn.relu(norm()(x))
    x = conv(self.n_features,
             kernel_size=(3, 3),
             strides=1,
             padding="SAME")(inp)
    x = norm()(x) + inp

    return nn.relu(x)


def shift_right(x: jax.Array, axis: int = 1) -> jax.Array:
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(x,
                   pad_widths,
                   mode="constant",
                   constant_values=x.dtype.type(0))
  
  return padded[:, :-1]


def sinusoidal_init(max_len: int = 256,
                    min_scale: float = 1.0,
                    max_scale: float = 1e4) -> Callable:
  def init(shape, dtype=jnp.float32) -> jax.Array:
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=dtype)
    position = np.arange(0, max_len)[:, None]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)

    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
    
    return jnp.array(pe[None, :, :]).astype(dtype)
  
  return init


class AddPosEmb(nn.Module):
  config: ModelConfig
  decode: bool = False

  @nn.compact
  def __call__(self, inp: jax.Array, inp_pos: Optional[jax.Array] = None):
    assert inp.ndim == 3
    config = self.config
  
    length = inp.shape[1]
    pos_emb_shape = (1, config.max_len, inp.shape[-1])
    pos_emb = sinusoidal_init(max_len=config.max_len)(pos_emb_shape, config.dtype)
    pe = pos_emb[:, :length, :]

    if self.decode:
      is_initialized = self.has_variable("cache", "cache_index")
      cache_index = self.variable("cache", "cache_index",
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, d_feature = pos_emb.shape
        pe = jax.lax.dynamic_slice(pos_emb,
                                   jnp.array((0, i, 0)),
                                   (1, 1, d_feature))
        
    if inp_pos is None:
      return inp + pe
    else:
      return inp + jnp.take(pe[0], inp_pos, axis=0)
    

class MLPBlock(nn.Module):
  config: ModelConfig
  out_features: Optional[int] = None

  @nn.compact
  def __call__(self, inp: jax.Array) -> jax.Array:
    config = self.config
    out_dim = self.out_features if self.out_features else inp.shape[-1]
    dense = partial(nn.Dense,
                    dtype=config.dtype,
                    kernel_init=config.kernel_init,
                    bias_init=config.bias_init)
    
    inp = dense(config.mlp_dim)(inp)
    inp = nn.Dropout(config.dropout_rate)(inp, config.deterministic)
    inp = nn.relu(inp)
    inp = dense(out_dim)(inp)
    inp = nn.Dropout(config.dropout_rate)(inp, config.deterministic)

    return inp
  

class Encoder1DBlock(nn.Module):
  config: ModelConfig

  @nn.compact
  def __call__(self, inp: jax.Array, enc_mask: Optional[jax.Array] = None):
    assert inp.ndim == 3
    config = self.config

    x = nn.LayerNorm(dtype=config.dtype)(inp)
    x = nn.SelfAttention(num_heads=config.num_heads,
                         dtype=config.dtype,
                         qkv_features=config.qkv_dim,
                         kernel_init=config.kernel_init,
                         use_bias=False,
                         broadcast_dropout=False,
                         dropout_rate=config.attention_dropout_rate,
                         deterministic=config.deterministic)(x, enc_mask)
    x = nn.Dropout(config.dropout_rate)(x, config.deterministic)
    x = inp + x

    inp = nn.LayerNorm(dtype=config.dtype)(x)
    inp = MLPBlock(config)(inp)

    return inp + x
  

class EncDec1DBlock(nn.Module):
  config: ModelConfig

  @nn.compact
  def __call__(self, targets: jax.Array, encoded: jax.Array,
               dec_mask: Optional[jax.Array] = None,
               enc_dec_mask: Optional[jax.Array] = None):
    assert targets.ndim == 3
    config = self.config

    x = nn.LayerNorm(dtype=config.dtype)(targets)
    x = nn.SelfAttention(num_heads=config.num_heads,
                         dtype=config.dtype,
                         qkv_features=config.qkv_dim,
                         kernel_init=config.kernel_init,
                         use_bias=False,
                         broadcast_dropout=False,
                         dropout_rate=config.attention_dropout_rate,
                         deterministic=config.deterministic,
                         decode=config.decode)(x, dec_mask)
    x = nn.Dropout(config.dropout_rate)(x, config.deterministic)
    x = x + targets

    y = nn.LayerNorm(dtype=config.dtype)(x)
    y = nn.MultiHeadDotProductAttention(
      num_heads=config.num_heads,
      dtype=config.dtype,
      qkv_features=config.qkv_dim,
      kernel_init=config.kernel_init,
      use_bias=False,
      broadcast_dropout=False,
      dropout_rate=config.attention_dropout_rate,
      deterministic=config.deterministic
    )(y, encoded, enc_dec_mask)
    y = nn.Dropout(config.dropout_rate)(y, config.deterministic)
    y = y + x

    x = nn.LayerNorm(dtype=config.dtype)(y)
    x = MLPBlock(config)(x)

    return x + y
