from .basic_types import Any, Tuple, List, DatasetList, DatasetItem, BatchType
from .utils import crop, Vocabulary

import jax.numpy as jnp
import os
import json
import cv2
import numpy as np
import random
from string import digits, ascii_letters
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool


class WPODSampler(object):
  def __init__(
      self,
      batch_size: int,
      ds_list: DatasetList,
      shuffle: bool = True,
      dtype: Any = jnp.float32,
  ) -> None:
    self._bs = batch_size
    self._ds = ds_list
    self._total = len(self._ds)
    self._cnt = 0
    self._shuffle = shuffle
    self._dtype = dtype

    self._cached = {}

    if shuffle:
      random.shuffle(self._ds)

  def sample(self, num: int = 1) -> BatchType:
    batch = self.ds[:num]
    with Pool(cpu_count() * 2) as p:
      batch = p.map(self._generate_data_point, batch)
      inps, labels = zip(*batch)

      return np.array(inps, dtype=self.dtype), np.array(labels, dtype=self.dtype)
    
  def _generate_data_point(self, ds_item: DatasetItem):
    if ds_item[0] in self._cached:
      return self._cached[ds_item[0]]
    else:
      img, coords, _ = load_image(ds_item)
      label = np.array(object_label(coords, 352, 1), dtype=np.float32)

      res = np.array(img / 255, dtype=np.float32), label
      self._cached[ds_item[0]] = res

      return res

  def __iter__(self):
    self._cnt = 0
    if self.shuffle:
      random.shuffle(self._ds)

    return self

  def __next__(self):
    if self.cnt + self.bs > self.total:
      raise StopIteration
    else:
      batch = self.ds[self.cnt:self.cnt + self.bs]
      self._cnt += self.bs

      with Pool(cpu_count() * 2) as p:
        batch = p.map(self._generate_data_point, batch)
        inps, labels = zip(*batch)

        return np.array(inps, dtype=self.dtype), np.array(labels, dtype=self.dtype)

  @property
  def bs(self):
    return self._bs

  @property
  def ds(self):
    return self._ds

  @property
  def total(self):
    return self._total

  @property
  def cnt(self):
    return self._cnt

  @property
  def shuffle(self):
    return self._shuffle

  @property
  def dtype(self):
    return self._dtype


class OCRSampler(object):
  def __init__(self, batch_size: int, ds_path: str, vocab: Vocabulary):
    self.batch_size = batch_size
    self._vocab = vocab
    self._ds = self._load(ds_path)

  def sample(self) -> Tuple[List[List[np.ndarray]], List[List[List[int]]]]:
    samples = []
    random.shuffle(self._ds)
    for i in range(len(self._ds) // self.batch_size):
      minibatch = self._ds[i * self.batch_size:(i + 1) * self.batch_size]
      img, label = zip(*minibatch)

      samples.append((list(img), self._vocab.batch_encode(list(label))))

    return zip(*samples)

  def _load(self, ds_path: str):
    files = os.listdir(ds_path)
    if_valid = (lambda x: (x[0] in self._vocab.keys) and
                (x[0] not in digits and x[0] not in ascii_letters) and
                (x.split("_")[0][-1] in self._vocab.keys))
    files = list(filter(if_valid, files))

    def load_image(filename: str) -> Tuple[np.ndarray, str]:
      img = np.flip(cv2.imread(os.path.join(ds_path, filename)), -1)
      label = filename.split("_")[0]

      img = cv2.resize(img, (144, 48)) / 255

      return img, label

    with Pool(cpu_count() * 2) as p:
      data = p.map(load_image, files)

    return data


def load_dataset(ds_dir: str, info_file: str):
  """Dataset item: (filename, normalized_plate_vertices_coords, plate_number)"""
  with open(os.path.join(ds_dir, info_file), "r") as fd:
    data = json.loads(fd.read())
  return [(os.path.join(ds_dir,
                        item["filename"]), item["coords"], item["number"])
          for item in data]


def load_image(ds_item) -> Tuple:
  img_pth, coords, number = ds_item

  img = np.flip(cv2.imread(img_pth), -1)
  coords = np.array(coords) * np.array([[img.shape[1]], [img.shape[0]]])

  img, coords = crop(img, coords.astype(int))
  coords = coords / img.shape[0]
  img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

  return img, coords, np.array(number)


def object_label(coords: np.ndarray, image_size: int, stride: int):
  scale = (image_size + 40) / 2 / stride
  size = image_size // stride
  label = np.zeros((size, size, 9))

  def _on_plate(x, y) -> bool:
    xs = coords[0]
    ys = coords[1]

    if not (xs.min() <= x <= xs.max() and ys.min() <= y <= ys.max()):
      return False
    res = False

    for i in range(4):
      j = 3 if i == 0 else i - 1
      if ((ys[i] > y) != (ys[j] > y)) and \
         (x < (xs[j] - xs[i]) * (y - ys[i]) / (ys[j] - ys[i]) + xs[i]):
        res = not res

    return res

  for i in range(size):
    y = (i + 0.5) / size
    for j in range(size):
      x = (j + 0.5) / size

      if _on_plate(x, y):
        label[i, j, 0] = 1
        points = coords * image_size / stride
        points = points - np.array([[j + 0.5], [i + 0.5]])
        points = points / scale
        label[i, j, 1:] = points.reshape((-1,))

  return label



if __name__ == "__main__":
  ds = load_dataset("data/train", "dataset.json")
  sampler = WPODSampler(32, ds)

  print(next(sampler))