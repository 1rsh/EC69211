import cv2
from utils import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("f", type=str)
parser.add_argument("-s", "--scale", type=float, default=-1)
parser.add_argument("-r", "--rotate", type=int, default=-1)
parser.add_argument("-t", "--type", type=str, default="bilinear")

class Exp1Image:
  def __init__(self, filepath):
    self.image = cv2.imread(filepath)
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    self.orig_image = self.image

  def scale(self, factor=2, _type="nearest", _return=False):
    if _return:
      return scale_image(self.image, k=factor, _type=_type)

    self.image = scale_image(self.image, k=factor, _type=_type)

  def rotate(self, angle=30, _type="nearest", _return=False):
    if _return:
      return rotate_image(self.image, theta=angle, _type=_type)

    self.image = rotate_image(self.image, theta=angle, _type=_type)

  def save(self, filepath):
    cv2.imwrite(filepath, cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
  os.makedirs("output/", exist_ok=True)

  args = vars(parser.parse_args())
  if args["scale"] == int(args["scale"]):
    args["scale"] = int(args["scale"])

  img = Exp1Image(args["f"][2:])

  if args["scale"] != -1:
    img.scale(factor=args["scale"], _type=args["type"])
    img.save(f"output/scale{args['scale']}_{args['type']}.png")

  if args["rotate"] != -1:
    img.rotate(angle=args["rotate"], _type=args["type"])
    img.save(f"output/rotate{args['rotate']}_{args['type']}.png")

  print("Saved image successfully")

  