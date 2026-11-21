#!/usr/bin/python
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="convert mnist <train>.csv, <test>.csv into <train>.dat, <test>.dat inputs for mnistasm")
parser.add_argument("--train", help="path to .csv file containing training set")
parser.add_argument("--test", help="path to .csv file containing test set")

args = parser.parse_args()

for group, path in (("train", args.train), ("test", args.test)):
  labels = []
  vectors = []

  with open(path) as f:
    for l in f:
      label = int(l[0])
      row = np.fromstring(l[2:], dtype=int, sep=",")
      im = row.reshape((28, 28))
      im = im[4:-4, 4:-4]
      vector = (im.flatten() > 127).tolist()

      labels.append(label)
      vectors.append(vector)

  with open(group + ".dat", "wb") as f:
    f.write(len(labels).to_bytes(4, byteorder="little"))
    f.write(bytes(labels))

    for vector in vectors:
      # permuting two vectors the same way doesn't change their dot product, so endianness doesn't matter
      bits = "".join("01"[b] for b in vector)
      block = int(bits, 2).to_bytes(64)
      f.write(block)
