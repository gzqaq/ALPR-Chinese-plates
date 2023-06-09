from JaxALPR.train import train_ocr
from JaxALPR.utils import set_random_seed, parse_flags_to_model_config, get_time

import jax
import os
from absl import app, flags
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_string("workdir", "ckpts", "Directory to store model data")
flags.DEFINE_string("restore_ckpt", "", "Directory that stores pre-trained checkpoints if want to restore")
flags.DEFINE_string("ds_dir", "dataset/ocr/train", "Train set directory")
flags.DEFINE_string("val_ds_dir", "dataset/ocr/val", "Validation set directory")
flags.DEFINE_integer("n_epochs", 100, "Number of training epochs")
flags.DEFINE_integer("train_batch_size", 128, "Training batch size")
flags.DEFINE_integer("val_batch_size", 64, "Validation batch size")
flags.DEFINE_float("clip_norm", 1.01, "Clip gradient norm")
flags.DEFINE_float("lr", 5e-5, "Learning rate")
config_flags.DEFINE_config_file("model",
                                "configs/model.py",
                                "Path to model config file",
                                lock_config=False)


def main(_):
  rng = set_random_seed(FLAGS.seed)
  train_config, eval_config, decode_config = parse_flags_to_model_config(
      FLAGS.model)

  FLAGS.model = train_config
  FLAGS.workdir = os.path.join(FLAGS.workdir, get_time())
  os.makedirs(FLAGS.workdir, exist_ok=True)

  rng, _rng = jax.random.split(rng)
  state, metrics = train_ocr(FLAGS, _rng)


if __name__ == "__main__":
  app.run(main)
