import cv2
import numpy as np
import sys
import math
from tqdm.auto import tqdm

def scale_image(image, k=2, _type = "nearest"):
  if type(k) != int:
    print(f"Scaling Factor = {k} of type: {type(k)} found. Might cause issues!", file=sys.stderr)

  height, width = image.shape[:2]
  new_height = int(height * k)
  new_width = int(width * k)
  scaled_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

  if _type == "nearest":
    for i in tqdm(range(new_height), leave=False, desc="Processing"):
      for j in range(new_width):
        src_i = int(i / k)
        src_j = int(j / k)
        scaled_image[i, j] = image[src_i, src_j]
  elif _type == "bilinear":
    for i in tqdm(range(new_height), leave=False, desc="Processing"):
      for j in range(new_width):
        i_floor = math.floor(i / k)
        j_floor = math.floor(j / k)
        i_ceil = min(height - 1, math.ceil(i / k))
        j_ceil = min(width - 1, math.ceil(j / k))

        x, y = i/k, j/k

        if (i_ceil == i_floor) and (j_ceil == j_floor):
          pixel = image[int(x), int(y), :]
        elif (i_ceil == i_floor):
          q1 = image[int(x), int(j_floor), :]
          q2 = image[int(x), int(j_ceil), :]
          pixel = q1 * (j_ceil - y) + q2 * (y - j_floor)
        elif (j_ceil == j_floor):
          q1 = image[int(i_floor), int(y), :]
          q2 = image[int(i_ceil), int(y), :]
          pixel = (q1 * (i_ceil - x)) + (q2	 * (x - i_floor))
        else:
          p1, p2, p3, p4 = image[i_floor, j_floor, :], image[i_ceil, j_floor, :], image[i_floor, j_ceil, :], image[i_ceil, j_ceil, :]

          int1 = p1 * (i_ceil - x) + p2 * (x - i_floor)
          int2 = p3 * (i_ceil - x) + p4 * (x - i_floor)
          pixel = int1 * (j_ceil - y) + int2 * (y - j_floor)

        scaled_image[i, j] = pixel
  else:
    print(f"Unrecognized type found! {_type}", file=sys.stderr)

  return scaled_image

def rotate_image(image, theta=30, _type="nearest"):
    if type(theta) not in [int, float]:
        print(f"Angle = {theta} of type: {type(theta)} found. Might cause issues!", file=sys.stderr)

    angle_rad = math.radians(theta)
    height, width = image.shape[:2]
    
    new_width = int(abs(width * math.cos(angle_rad)) + abs(height * math.sin(angle_rad)))
    new_height = int(abs(height * math.cos(angle_rad)) + abs(width * math.sin(angle_rad)))
    
    new_center_x = new_width / 2
    new_center_y = new_height / 2
    old_center_x = width / 2
    old_center_y = height / 2

    rotated_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

    if _type == "nearest":
        for i in tqdm(range(new_height), leave=False, desc="Processing"):
            for j in range(new_width):
                x = (j - new_center_x) * math.cos(angle_rad) + (i - new_center_y) * math.sin(angle_rad) + old_center_x
                y = -(j - new_center_x) * math.sin(angle_rad) + (i - new_center_y) * math.cos(angle_rad) + old_center_y
                x = int(x)
                y = int(y)
                if 0 <= x < width and 0 <= y < height:
                    rotated_image[i, j] = image[y, x]
    elif _type == "bilinear":
        for i in tqdm(range(new_height), leave=False, desc="Processing"):
            for j in range(new_width):
                x = (j - new_center_x) * math.cos(angle_rad) + (i - new_center_y) * math.sin(angle_rad) + old_center_x
                y = -(j - new_center_x) * math.sin(angle_rad) + (i - new_center_y) * math.cos(angle_rad) + old_center_y

                x0 = int(np.floor(x))
                x1 = x0 + 1
                y0 = int(np.floor(y))
                y1 = y0 + 1

                if 0 <= x0 < width and 0 <= y0 < height:
                    I00 = image[y0, x0] if 0 <= x0 < width and 0 <= y0 < height else np.array([0, 0, 0], dtype=np.uint8)
                    I01 = image[y0, x1] if 0 <= x1 < width and 0 <= y0 < height else np.array([0, 0, 0], dtype=np.uint8)
                    I10 = image[y1, x0] if 0 <= x0 < width and 0 <= y1 < height else np.array([0, 0, 0], dtype=np.uint8)
                    I11 = image[y1, x1] if 0 <= x1 < width and 0 <= y1 < height else np.array([0, 0, 0], dtype=np.uint8)

                    alpha = x - x0
                    beta = y - y0

                    interpolated_pixel = (I00 * (1 - alpha) * (1 - beta) +
                                          I01 * alpha * (1 - beta) +
                                          I10 * (1 - alpha) * beta +
                                          I11 * alpha * beta)

                    rotated_image[i, j] = np.clip(interpolated_pixel, 0, 255).astype(np.uint8)
    else:
        print(f"Unrecognized type found! {_type}", file=sys.stderr)

    return rotated_image