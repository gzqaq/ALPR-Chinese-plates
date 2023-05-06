from .models.ocr import OCR
from .utils import Vocabulary

import jax.numpy as jnp
import numpy as np
import torch
from flax.core.frozen_dict import FrozenDict
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from typing import Sequence, Dict


def generate_mask(
    model: CLIPSegForImageSegmentation,
    processor: CLIPSegProcessor,
    images: Sequence[np.ndarray],
    threshold: float = 0.7,
) -> np.ndarray:
  if images[0].ndim == 2:
    images = torch.tensor(images[None, ...]).to(model.device)
  else:
    images = torch.tensor(images).to(model.device)

  inputs = processor(
      text=["license plate"] * len(images),
      images=images,
      padding="max_length",
      return_tensors="pt",
  ).to(model.device)

  with torch.no_grad():
    logits = model(**inputs).logits

  mask = torch.sigmoid(logits).cpu().numpy()[..., None] > threshold

  return mask


def crop_image_by_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
  y_min, *_, y_max = mask.any(axis=1).nonzero()[0] / mask.shape[0]
  x_min, *_, x_max = mask.any(axis=0).nonzero()[0] / mask.shape[1]

  w, h = image.size
  return image.crop((x_min * w, y_min * h, x_max * w, y_max * h))


def detect(
    images: Sequence[np.ndarray],
    model: CLIPSegForImageSegmentation,
    processor: CLIPSegProcessor,
    threshold: float = 0.7,
) -> Sequence[np.ndarray]:
  if images[0].ndim == 2:
    images = [images]
  plates = []

  for image in images:
    mask = generate_mask(model, processor, image, threshold)
    cropped = crop_image_by_mask(Image.fromarray(image), mask)

    # Inference twice
    cropped = crop_image_by_mask(
        cropped, generate_mask(model, processor, np.array(cropped), threshold))

    plates.append(np.array(cropped.resize((144, 48))))

  if len(plates) == 1:
    plates = plates[0]

  return plates


def recognize(
    plate_images: Sequence[np.ndarray],
    decode_model: OCR,
    variables: Dict[str, FrozenDict],
    vocab: Vocabulary,
):
  if plate_images[0].ndim == 2:
    plate_images = [plate_images]

  encoded = decode_model.apply(
      variables,
      jnp.array(plate_images, dtype=decode_model.config.dtype) / 255,
      method=decode_model.encode,
  )

  cache = variables
  decoded = []
  targets = jnp.zeros([encoded.shape[0], 1])

  for _ in range(9):
    logits, cache = decode_model.apply(
        {
            "params": variables["params"],
            "cache": cache["cache"]
        },
        encoded,
        None,
        targets,
        mutable=["cache"],
        method=decode_model.decode,
    )
    targets = logits.argmax(axis=-1)
    decoded.append(vocab.batch_decode(targets.tolist()))

  decoded = list(zip(*decoded))  # 9 x bs x 1 -> bs x 9 x 1

  concat_str = lambda lst: "".join(list(map(lambda x: x[0], lst)))[:8]
  return list(map(lambda x: x.split("O")[0], map(concat_str, decoded)))
