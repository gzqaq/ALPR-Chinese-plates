from JaxALPR.train import train_wpod
from JaxALPR.utils import set_random_seed, parse_flags_to_model_config

import jax
from absl import app, flags
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_string("ds_dir", "data/train", "Dataset directory")
flags.DEFINE_string("ds_info_file", "dataset.json", "Name of json dataset file")
flags.DEFINE_string("dataset_split", "7-2-1", "Train set, validation set and test set")
flags.DEFINE_integer("n_epochs", 100, "Number of training epochs")
flags.DEFINE_integer("train_batch_size", 64, "Training batch size")
flags.DEFINE_integer("val_batch_size", 64, "Validation batch size")
flags.DEFINE_float("clip_norm", 1.01, "Clip gradient norm")
flags.DEFINE_float("lr", 5e-5, "Learning rate")
config_flags.DEFINE_config_file("model", "configs/wpod.py", "Path to model config file", lock_config=False)


def main(_):
  rng = set_random_seed(FLAGS.seed)
  train_config, eval_config, decode_config = parse_flags_to_model_config(FLAGS)

  FLAGS.model = train_config

  rng, _rng = jax.random.split(rng)
  state, metrics = train_wpod(FLAGS, _rng)


if __name__ == "__main__":
  app.run(main)