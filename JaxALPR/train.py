from .basic_types import Any, Tuple, BatchType, KeyArray, MetricType
from .models.detect import wpod_loss, WPOD
from .models.ocr import OCR
from .dataset import load_dataset, WPODSampler, OCRSampler
from .utils import Vocabulary

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from flax.training import train_state, checkpoints
from ml_collections import ConfigDict


class TrainState(train_state.TrainState):
  batch_stats: Any

def train_ocr(config: ConfigDict, rng: KeyArray) -> Tuple[TrainState, MetricType]:
  def _update_minibatch(runner_state: Tuple[TrainState, KeyArray],
                        batch: BatchType):
    state, rng = runner_state
    inps, labels = batch

    rng, dropout_rng = jax.random.split(rng)
    def loss_fn(params):
      logits, new_model_state = state.apply_fn(
        {"params": params, "batch_stats": state.batch_stats},
        inps, labels,
        mutable=["batch_stats"],
        rngs={"dropout": dropout_rng}
      )

      onehot = jax.nn.one_hot(labels, config.model.vocab_size, dtype=config.model.dtype)
      loss = optax.softmax_cross_entropy(logits, onehot).sum()

      return loss, (new_model_state,)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_vals), grad = grad_fn(state.params)
    new_model_state, *_ = aux_vals

    new_state = state.apply_gradients(
      grads=grad,
      batch_stats=new_model_state["batch_stats"])
    train_metrics = {"loss": loss}

    return (new_state, rng), train_metrics
  
  def _evaluate(variables, batch):
    inps, labels = batch
    logits = OCR(config.model.replace(deterministic=True)).apply(
      variables, inps, labels, mutable=False
    )

    onehot = jax.nn.one_hot(labels, config.model.vocab_size, dtype=config.model.dtype)
    loss = optax.softmax_cross_entropy(logits, onehot).sum()

    return loss
  
  model = OCR(config.model)
  dummy_inp = jnp.ones([4, 48, 144, 3])
  targets = jnp.ones([4, 9])
  rng, init_rng, dropout_rng = jax.random.split(rng, 3)
  variables = model.init({"params": init_rng, "dropout": dropout_rng},
                         dummy_inp, targets)
  params = variables["params"]
  batch_stats = variables["batch_stats"]

  tx = optax.chain(optax.clip_by_global_norm(config.clip_norm),
                   optax.adam(config.lr))
  state = TrainState.create(apply_fn=model.apply,
                            params=params,
                            tx=tx,
                            batch_stats=batch_stats)
  del model, variables, params, batch_stats

  total_metrics = {"train_loss": [], "val_loss": []}
  train_sampler = OCRSampler(config.train_batch_size)
  val_sampler = OCRSampler(config.val_batch_size)
  vocab = Vocabulary(9)
  logging.info("Start training...")
  for i_epoch in range(config.n_epochs):
    inps, labels = train_sampler.sample(config.n_minibatches, vocab)
    minibatches = jnp.array(inps, config.model.dtype), jnp.array(labels)

    runner_state, metrics = jax.lax.scan(_update_minibatch, (state, rng), minibatches)
    state, rng = runner_state
    total_metrics["train_loss"].append(metrics["loss"])

    val_inp, val_labels = val_sampler.sample(1, vocab)
    val_batch = jnp.array(val_inp[0], config.model.dtype), jnp.array(val_labels[0])
    total_metrics["val_loss"].append(jax.jit(_evaluate)({"params": state.params, "batch_stats": state.batch_stats}, val_batch))

    logging.info("Epoch %d: train_loss %f, val_loss %f", i_epoch, np.mean(metrics["loss"]), total_metrics["val_loss"][-1])
    checkpoints.save_checkpoint(config.workdir, state, state.step, keep=3)

  return state, total_metrics


def train_wpod(config: ConfigDict, rng: KeyArray) -> Tuple[TrainState, MetricType]:
  def _update_minibatch(runner_state: Tuple[TrainState, KeyArray],
                        batch: BatchType):
    state, rng = runner_state
    inps, labels = batch

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

    return (new_state, rng), train_metrics
  
  def _evaluate(state: TrainState, val_batch: BatchType):
    val_inps, val_labels = val_batch
    logits = state.apply_fn(
      {"params": state.params, "batch_stats": state.batch_stats},
      val_inps, False,
      mutable=False
    )

    return wpod_loss(logits, val_labels)
  
  model = WPOD(config.model)
  ds = load_dataset(config.ds_dir, config.ds_info_file)
  logging.info("Successfully load dataset! (%d images)", len(ds))
  ds_split = list(map(lambda x: int(x), config.dataset_split.split("-")))

  unit = len(ds) // sum(ds_split)
  train_split = unit * ds_split[0]
  val_split = unit * ds_split[1] + train_split
  train_sampler = WPODSampler(config.train_batch_size, ds[:train_split], True, config.model.dtype)
  val_sampler = WPODSampler(config.val_batch_size, ds[train_split:val_split], True, config.model.dtype)

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

  total_metrics = {"train_loss": [], "val_loss": []}
  logging.info("Start training...")
  val_iter = iter(val_sampler)
  for i_epoch in range(config.n_epochs):
    train_loss = []
    for batch in iter(train_sampler):
      (state, rng), metrics = jax.jit(_update_minibatch)((state, rng), batch)
      train_loss.append(metrics["loss"].item())

    try:
      batch = next(val_iter)
    except StopIteration:
      val_iter = iter(val_sampler)
      batch = next(val_iter)
    val_loss = jax.jit(_evaluate)(state, batch).item()

    logging.info("Epoch %d: train_loss %f, val_loss %f", i_epoch, np.mean(train_loss), val_loss)
    total_metrics["train_loss"].append(train_loss)
    total_metrics["val_loss"].append(val_loss)

    checkpoints.save_checkpoint(config.workdir, state, state.step, keep=3)

  return state, metrics