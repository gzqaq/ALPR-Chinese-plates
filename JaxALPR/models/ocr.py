from . import ModelConfig, ResBlock, AddPosEmb, Encoder1DBlock, EncDec1DBlock, shift_right

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Optional


class ImageEmbed(nn.Module):
  config: ModelConfig
  out_features: int

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
    max_pool = partial(nn.max_pool,
                       window_shape=(2, 2),
                       strides=(2, 2),
                       padding="VALID")
    
    inp = conv(features=self.out_features // 8,
             kernel_size=(7, 7),
             strides=2,
             padding="SAME")(inp)
    inp = nn.relu(norm()(inp))

    for i in [8, 4, 2]:
      inp = conv(self.out_features // i, (3, 3), 1, "SAME")(inp)
      inp = ResBlock(config, self.out_features // i)(inp, train)
      inp = max_pool(norm()(inp))

    inp = conv(self.out_features, (3, 3), 1, "SAME")(inp)
    inp = ResBlock(config, self.out_features)(inp, train)
    inp = nn.relu(norm()(inp))

    inp = jnp.einsum("bhwc->bcwh", inp)
    inp = jnp.reshape(inp, (inp.shape[0], inp.shape[1], -1))

    return jnp.einsum("bck->bkc", inp)
  

class Encoder(nn.Module):
  config: ModelConfig

  @nn.compact
  def __call__(self, inp: jax.Array,
               inp_pos: Optional[jax.Array] = None,
               enc_mask: Optional[jax.Array] = None):
    assert inp.ndim == 4
    config = self.config
    train = not config.deterministic

    img_emb = ImageEmbed(config, config.embed_dim, name="img_emb")(inp, train)
    inp = AddPosEmb(config, decode=False, name="pos_emb")(img_emb, inp_pos)
    inp = nn.Dropout(config.dropout_rate)(inp, config.deterministic)
    inp = inp.astype(config.dtype)

    for lyr in range(config.num_layers):
      inp = Encoder1DBlock(config, name=f"enc_block_{lyr}")(inp, enc_mask)

    inp = nn.LayerNorm(dtype=config.dtype, name="enc_norm")(inp)

    return inp
  

class Decoder(nn.Module):
  config: ModelConfig

  @nn.compact
  def __call__(self, encoded: jax.Array, targets: jax.Array,
               targets_pos: Optional[jax.Array] = None,
               dec_mask: Optional[jax.Array] = None,
               enc_dec_mask: Optional[jax.Array] = None):
    assert encoded.ndim == 3   # (bs, len, depth)
    assert targets.ndim == 2   # (bs, len)

    config = self.config

    output_emb = nn.Embed(num_embeddings=config.vocab_size,
                          features=config.embed_dim,
                          embedding_init=nn.initializers.normal(1))
    y = targets.astype("int32")
    if not config.decode:
      y = shift_right(y)
    y = output_emb(y)
    y = AddPosEmb(config, config.decode, name="pos_emb_out")(y, targets_pos)
    y = nn.Dropout(config.dropout_rate)(y, config.deterministic)
    y = y.astype(config.dtype)

    for lyr in range(config.num_layers):
      y = EncDec1DBlock(config, name=f"enc_dec_block_{lyr}")(y,
                                                             encoded,
                                                             dec_mask,
                                                             enc_dec_mask)
    y = nn.LayerNorm(dtype=config.dtype, name="enc_dec_norm")(y)

    if config.logits_via_embedding:
      logits = output_emb.attend(y.astype(jnp.float32))
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = nn.Dense(features=config.vocab_size,
                        dtype=config.dtype,
                        kernel_init=config.kernel_init,
                        bias_init=config.bias_init,
                        name="logit_dense")(y)
      
    return logits
  

class OCR(nn.Module):
  config: ModelConfig

  def setup(self):
    config = self.config

    self.encoder = Encoder(config)
    self.decoder = Decoder(config)

  def encode(self, inp: jax.Array,
             inp_pos: Optional[jax.Array] = None,
             inp_seg: Optional[jax.Array] = None):
    # config = self.config

    # enc_mask = nn.make_attention_mask(inp > 0, inp > 0, dtype=config.dtype)

    # if inp_seg is not None:
    #   enc_mask = nn.make_attention_mask(
    #     enc_mask,
    #     nn.make_attention_mask(inp_seg,
    #                            inp_seg,
    #                            jnp.equal,
    #                            dtype=config.dtype))
      
    return self.encoder(inp, None, None)
  
  def decode(self, encoded: jax.Array, inp: jax.Array, targets: jax.Array,
             targets_pos: Optional[jax.Array] = None,
             inp_seg: Optional[jax.Array] = None,
             targets_seg: Optional[jax.Array] = None):
    config = self.config
    inp_mask = jnp.ones(encoded.shape[:2]) > 0

    if config.decode:
      dec_mask = None
      enc_dec_mask = nn.make_attention_mask(jnp.ones_like(targets) > 0,
                                            inp_mask,
                                            dtype=config.dtype)
    else:
      dec_mask = nn.combine_masks(nn.make_attention_mask(targets > 0,
                                                         targets > 0,
                                                         dtype=config.dtype),
                                  nn.make_causal_mask(targets,
                                                      dtype=config.dtype))
      enc_dec_mask = nn.make_attention_mask(targets > 0,
                                            inp_mask,
                                            dtype=config.dtype)
      
    if inp_seg is not None:
      dec_mask = nn.combine_masks(dec_mask,
                                  nn.make_attention_mask(targets_seg,
                                                         targets_seg,
                                                         jnp.equal,
                                                         dtype=config.dtype))
      enc_dec_mask = nn.combine_masks(
        enc_dec_mask,
        nn.make_attention_mask(targets_seg,
                               inp_seg,
                               jnp.equal,
                               dtype=config.dtype))
      
    logits = self.decoder(encoded, targets,
                          targets_pos, dec_mask, enc_dec_mask)
    return logits.astype(config.dtype)
  
  def __call__(self, inp: jax.Array, targets: jax.Array,
               inp_pos: Optional[jax.Array] = None,
               targets_pos: Optional[jax.Array] = None,
               inp_seg: Optional[jax.Array] = None,
               targets_seg: Optional[jax.Array] = None):
    encoded = self.encode(inp, inp_pos, inp_seg)
    return self.decode(encoded, inp, targets,
                       targets_pos, inp_seg, targets_seg)


if __name__ == "__main__":
  config = ModelConfig(deterministic=True)
  rng = jax.random.PRNGKey(7)

  rng, _rng = jax.random.split(rng)
  dummy_inp = jax.random.uniform(_rng, [4, 48, 144, 3])
  targets = jnp.ones((4, 10))

  # img_emb = ImageEmbed(config, config.embed_dim)
  # rng, _rng = jax.random.split(rng)
  # variables = img_emb.init(_rng, dummy_inp)

  # output = img_emb.apply(variables, dummy_inp, False)

  # print(output.shape)

  ocr = OCR(config)
  rng, init_rng, dropout_rng = jax.random.split(rng, 3)
  variables = ocr.init({"params": init_rng, "dropout": dropout_rng},
                       dummy_inp, targets)

  rng, _rng = jax.random.split(rng)
  output = ocr.apply(variables, dummy_inp,
                     targets,
                     rngs={"dropout": _rng})
  
  print(output.shape)