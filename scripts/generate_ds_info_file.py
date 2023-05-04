import json
import numpy as np
import os
from absl import app, flags, logging

H = 1160
W = 720
FLAGS = flags.FLAGS

flags.DEFINE_string("ds_dir", "data/train", "Directory of dataset")
flags.DEFINE_string("filename", "dataset.json", "Name of info file")


def main(_):
  files = os.listdir(FLAGS.ds_dir)
  files = list(filter(lambda x: x[-3:] == "jpg", files))
  logging.info("Find %d images.", len(files))

  ds = []

  for f in files:
    area, tilt, bb, lp, lpn, bright, blur = f.split("-")
    bb_coords = bb.split("_")
    lp_coords = lp.split("_")
    parse_coords = lambda x: [int(x[0]) / W, int(x[1]) / H]
    bb_coords = list(map(lambda x: parse_coords(x.split("&")), bb_coords))
    lp_coords = list(map(lambda x: parse_coords(x.split("&")), lp_coords))
    lpn = list(map(lambda x: int(x), lpn.split("_")))

    ds.append({"filename": f, "coords": np.array(lp_coords).T.tolist(), "number": lpn})

  with open(os.path.join(FLAGS.ds_dir, FLAGS.filename), "w") as fd:
    json.dump(ds, fd)


if __name__ == "__main__":
  app.run(main)