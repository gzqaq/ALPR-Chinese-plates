from configs.model import get_config
from JaxALPR.basic_types import KeyArray
from JaxALPR.inference import detect, recognize
from JaxALPR.models.ocr import OCR
from JaxALPR.train import TrainState
from JaxALPR.utils import set_random_seed, parse_flags_to_model_config, Vocabulary

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import torch
from absl import app, flags, logging
from flax.training import checkpoints
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_string(
    "clipseg_ckpt_dir",
    "ckpts/clipseg",
    "Path to the directory storing CLIPSeg checkpoint",
)
flags.DEFINE_string("ocr_ckpt_dir", "ckpts/ocr",
                    "Path to the directory storing OCR checkpoint")
flags.DEFINE_integer(
    "batch_size", 3,
    "Inference batch size (number of images to recognize one time)")
flags.DEFINE_integer("gpu", -1, "Which GPU to run inference (use CPU if -1)")
flags.DEFINE_bool("save_plate", False, "Save extracted plates.")
flags.DEFINE_float("thres", 0.7, "Threshold to generate mask")


def load_model(rng: KeyArray):
  logging.info("Loading models...")

  # Detection
  device = (torch.device("cpu")
            if FLAGS.gpu == -1 else torch.device(f"cuda:{FLAGS.gpu}"))
  processor = CLIPSegProcessor.from_pretrained(FLAGS.clipseg_ckpt_dir)
  clipseg = CLIPSegForImageSegmentation.from_pretrained(
      FLAGS.clipseg_ckpt_dir).to(device)

  logging.info("Detection model loaded.")

  # OCR
  *_, decode_config = parse_flags_to_model_config(get_config())

  ocr = OCR(decode_config)
  rng, _rng = jax.random.split(rng)
  variables = ocr.init(_rng, jnp.ones([FLAGS.batch_size, 48, 144, 3]),
                       jnp.ones([FLAGS.batch_size, 9]))

  tx = optax.chain(optax.clip_by_global_norm(1.01), optax.adam(5e-5))
  state = TrainState.create(
      apply_fn=ocr.apply,
      params=variables["params"],
      tx=tx,
      batch_stats=variables["batch_stats"],
  )
  state = checkpoints.restore_checkpoint(FLAGS.ocr_ckpt_dir, state)
  decode_variables = {
      "params": state.params,
      "batch_stats": state.batch_stats,
      "cache": variables["cache"],
  }
  vocab = Vocabulary(10)

  logging.info("OCR model loaded.")

  return {
      "detect": {
          "model": clipseg,
          "processor": processor
      },
      "ocr": {
          "decode_model": ocr,
          "variables": decode_variables,
          "vocab": vocab
      },
  }


def run_interactive(models):
  detect_model = models["detect"]
  ocr_model = models["ocr"]

  print(
      "Enter paths to image or directory to recognize (separated by space), or 'quit' to end the program."
  )
  try:
    while True:
      comm = input(">>> ")

      if comm == "quit":
        return
      else:
        pths = comm.split(" ")
        img_pths = []
        for pth in pths:
          if not os.path.exists(pth):
            logging.warning("%s doesn't exist!", pth)
            continue
          if os.path.isfile(pth):
            img_pths.append(pth)
          elif os.path.isdir(pth):
            files = os.listdir(pth)
            files = list(map(lambda x: os.path.join(pth, x), files))
            img_pths.extend(files)

        imgs = list(map(lambda pth: np.array(Image.open(pth)), img_pths))
        plates = detect(imgs, **detect_model, threshold=FLAGS.thres)

        if FLAGS.save_plate:
          for img_pth, plate in zip(img_pths, plates):
            save_pth = img_pth[:-4] + "_plate.jpg"
            Image.fromarray(plate).save(save_pth)
            logging.info("Saving plate: %s", save_pth)

        numbers = recognize(plates, **ocr_model)

        for pth, number in zip(img_pths, numbers):
          print(f"{pth}: {number}")
  except KeyboardInterrupt:
    return


def main(_):
  rng = set_random_seed(FLAGS.seed)

  rng, _rng = jax.random.split(rng)
  models = load_model(_rng)

  run_interactive(models)


if __name__ == "__main__":
  app.run(main)
