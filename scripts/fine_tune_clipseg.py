from JaxALPR.dataset import WPODSampler, load_dataset

import torch
import numpy as np

from absl import app, flags, logging
from transformers import AutoProcessor, CLIPSegForImageSegmentation

FLAGS = flags.FLAGS

flags.DEFINE_string("ckpt_path", "ckpts/clipseg", "Directory path to saved model")
flags.DEFINE_string("ds_path", "data", "Directory path to train data set")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("n_epochs", 100, "Number of training epochs")
flags.DEFINE_integer("save_period", 3, "Number of epochs between checkpoint saving")
flags.DEFINE_integer("gpu", 0, "Which GPU to train on")


def train_loop(model, optim, processor):
  ds = load_dataset(FLAGS.ds_path, "dataset.json")
  sampler = WPODSampler(FLAGS.batch_size, ds)

  train_losses = []
  for i in range(FLAGS.n_epochs):
    train_loss = []
    for batch in iter(sampler):
      inps, labels = batch
      inps = np.einsum("bhwc->bchw", inps)
      labels = torch.tensor(labels[..., 0]).to(model.device)
      model_inputs = processor(text=["license plate"] * inps.shape[0],
                               images=inps,
                               padding="max_length",
                               return_tensors="pt").to(model.device)
      
      logits = model(**model_inputs).logits
      loss = torch.binary_cross_entropy_with_logits(logits, labels).sum()

      optim.zero_grad()
      loss.backward()
      optim.step()
      train_loss.append(loss.item())

    train_losses.append(train_loss)
    logging.info("Epoch %d: train_loss %f", i, np.mean(train_loss))

    if (i + 1) % FLAGS.save_period == 0:
      model.save_pretrained(FLAGS.ckpt_path)
      logging.info("Save checkpoint to %s", FLAGS.ckpt_path)

  return model, optim


def main(_):
  device = torch.device(f"cuda:{FLAGS.gpu}")

  processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
  model = CLIPSegForImageSegmentation.from_pretrained(FLAGS.ckpt_path).to(device)
  optim = torch.optim.AdamW(model.parameters(), 5e-5)

  logging.info("Start training...")
  model, optim = train_loop(model, optim, processor)

  model.save_pretrained(FLAGS.ckpt_path)
  logging.info("Save final checkpoint to %s", FLAGS.ckpt_path)


if __name__ == "__main__":
  app.run(main)