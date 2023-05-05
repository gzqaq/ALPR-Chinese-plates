from . import ModelConfig, ResBlock

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial


ZERO = 1e-6


class DetectBlock(nn.Module):
  config: ModelConfig

  @nn.compact
  def __call__(self, inp: jax.Array):
    assert inp.ndim == 4

    config = self.config
    conv = partial(nn.Conv,
                   dtype=config.dtype,
                   kernel_init=config.kernel_init,
                   use_bias=False)

    clsf = conv(features=2,
                kernel_size=(3, 3),
                strides=1,
                padding="SAME")(inp)
    clsf = nn.softmax(clsf, axis=-1)
    affine = conv(features=6,
                  kernel_size=(3, 3),
                  strides=1,
                  padding="SAME")(inp)

    return jnp.concatenate([clsf, affine], axis=-1)


class WPOD(nn.Module):
  config: ModelConfig

  @nn.compact
  def __call__(self, inp: jax.Array, train: bool = True) -> jax.Array:
    config = self.config
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=config.dtype)
    conv = partial(nn.Conv,
                   kernel_size=(3, 3),
                   strides=1,
                   padding="SAME",
                   dtype=config.dtype,
                   kernel_init=config.kernel_init,
                   use_bias=False)
    max_pool = partial(nn.max_pool,
                       window_shape=(2, 2),
                       strides=(2, 2),
                       padding="VALID")

    for _ in range(2):
      inp = nn.relu(norm()(conv(16)(inp)))
    inp = max_pool(inp)

    inp = nn.relu(norm()(conv(32)(inp)))
    inp = ResBlock(config, 32)(inp, train)
    inp = max_pool(norm()(inp))

    inp = nn.relu(norm()(conv(64)(inp)))
    for _ in range(2):
      inp = norm()(ResBlock(config, 64)(inp, train))
    inp = max_pool(inp)
    for _ in range(2):
      inp = norm()(ResBlock(config, 64)(inp, train))
    inp = max_pool(inp)

    inp = nn.relu(norm()(conv(128)(inp)))
    for _ in range(2):
      inp = norm()(ResBlock(config, 128)(inp, train))

    return DetectBlock(config)(inp)


@jax.jit
def wpod_loss(pred: jax.Array, label: jax.Array) -> jax.Array:
  obj_pred = pred[..., 0]
  bg_pred = pred[..., 1]
  obj_label = label[..., 0]
  bg_label = 1 - obj_label

  def log_loss(pred, label):
    loss = -label * jnp.log(jnp.clip(pred, ZERO, 1.))
    return jnp.mean(loss, axis=0)

  prob_loss = log_loss(obj_pred, obj_label) + log_loss(bg_pred, bg_label)

  affine_1 = jnp.concatenate([nn.relu(pred[..., 2:3]),
                              pred[..., 3:4],
                              pred[..., 4:5]], axis=-1)
  affine_2 = jnp.concatenate([pred[..., 5:6],
                              nn.relu(pred[..., 6:7]),
                              pred[..., 7:8]], axis=-1)
  affine = jnp.concatenate([affine_1, affine_2], axis=-1)

  unit = jnp.array([[-.5, -.5, 1],
                    [.5, -.5, 1],
                    [.5, .5, 1],
                    [-.5, .5, 1]])
  unit = jnp.tile(unit[None, None, None, ...], pred.shape[:3] + (1, 1))
  points_pred = jnp.einsum("bhwik,bhwjk->bhwij",
                           affine.reshape(affine.shape[:3] + (2, 3)),
                           unit).reshape(affine.shape[:3] + (-1,))
  points_label = label[..., 1:]
  mask = obj_label[..., None]
  affine_loss = jnp.linalg.norm(points_label * mask - points_pred * mask,
                                ord=1, axis=(0, -1))

  return jnp.sum(prob_loss + affine_loss)


if __name__ == "__main__":
  rng = jax.random.PRNGKey(7)
  wpod = WPOD(ModelConfig())

  rng, _rng = jax.random.split(rng)
  dummy_inp = jax.random.uniform(_rng, [1, 256, 256, 3])

  rng, _rng = jax.random.split(rng)
  variables = wpod.init(_rng, dummy_inp)

  output = wpod.apply(variables, dummy_inp, False)

  rng, _rng = jax.random.split(rng)
  fake_labels = jax.random.uniform(_rng, [1, 16, 16, 9])

  print(wpod_loss(output, fake_labels))
