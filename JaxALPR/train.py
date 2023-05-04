from .basic_types import Any, Tuple, BatchType, KeyArray, MetricType
from .models.detect import wpod_loss, WPOD
from .dataset import load_dataset, WPODSampler

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from flax.training import train_state
from ml_collections import ConfigDict


class TrainState(train_state.TrainState):
  batch_stats: Any


def train_wpod(config: ConfigDict, rng: KeyArray) -> Tuple[TrainState, MetricType]:
  def _update_minibatch(runner_state: Tuple[TrainState, KeyArray],
                        batch: Tuple[BatchType, BatchType]):
    state, rng = runner_state
    train_batch, val_batch = batch
    inps, labels = train_batch
    val_inps, val_labels = val_batch

    rng, dropout_rng = jax.random.split(rng)
    def loss_fn(params):
      logits, new_model_state = state.apply_fn(
        {"params": params, "batch_stats": state.batch_stats},
        inps, True,
        mutable=["batch_stats"],
        rngs={"dropout": dropout_rng}
      )

      return wpod_loss(logits, labels), (new_model_state,)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_vals), grad = grad_fn(state.params)
    new_model_state, *_ = aux_vals

    new_state = state.apply_gradients(
      grads=grad,
      batch_stats=new_model_state["batch_stats"]
    )
    train_metrics = {"loss": loss}

    def evaluate(params):
      logits = new_state.apply_fn(
        {"params": params, "batch_stats": new_state.batch_stats},
        val_inps, False,
        mutable=False,
      )

      return wpod_loss(logits, val_labels)
    val_metrics = {"loss": evaluate(new_state.params)}

    return (new_state, rng), {"train": train_metrics, "val": val_metrics}
  
  model = WPOD(config.model)
  ds = load_dataset(config.ds_dir, config.ds_info_file)
  ds_split = list(map(lambda x: int(x), config.dataset_split.split("-")))

  unit = len(ds) // sum(ds_split)
  train_split = unit * ds_split[0]
  val_split = unit * ds_split[1] + train_split
  train_sampler = WPODSampler(config.train_batch_size, ds[:train_split], True, config.model.dtype)
  val_sampler = WPODSampler(config.val_batch_size, ds[train_split:val_split], True, config.model.dtype)
  test_sampler = WPODSampler(config.val_batch_size, ds[val_split:], False, config.model.dtype)

  sample_inps, _ = train_sampler.sample(1)
  rng, init_rng, dropout_rng = jax.random.split(rng, 3)
  variables = model.init({"params": init_rng, "dropout": dropout_rng},
                         sample_inps)
  params = variables["params"]
  batch_stats = variables["batch_stats"]

  tx = optax.chain(optax.clip_by_global_norm(config.clip_norm),
                   optax.adamw(config.lr))

  state = TrainState.create(apply_fn=model.apply,
                            params=params,
                            tx=tx,
                            batch_stats=batch_stats)
  del model, variables, params, batch_stats

  metrics = {"train_loss": [], "val_loss": []}
  for i_epoch in range(config.n_epochs):
    train_loss = []
    val_loss = []

    val_iter = iter(val_sampler)
    for train_batch in iter(train_sampler):
      try:
        val_batch = next(val_iter)
      except StopIteration:
        val_iter = iter(val_sampler)
        val_batch = next(val_iter)

      (state, rng), metrics = jax.jit(_update_minibatch)((state, rng), (train_batch, val_batch))
      train_loss.append(metrics["train"]["loss"])
      val_loss.append(metrics["val"]["loss"])

    train_loss = np.mean(train_loss)
    val_loss = np.mean(val_loss)
    logging.info("Epoch %d: train_loss %f, val_loss %f", i_epoch, train_loss, val_loss)
    metrics["train_loss"].append(train_loss)
    metrics["val_loss"].append(val_loss)

  return state, metrics