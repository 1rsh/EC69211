import matplotlib.pyplot as plt
import numpy as np

class BMP:
  def __init__(self):
    return

  def read(self, filepath, empty=False):
    assert filepath.endswith(('.bmp')), "Extension must be '.bmp'!"
    self.filepath = filepath
    self.image = self.read_image(self.filepath)

    self.header = self.image[:14]
    self.info_header = self.image[14:54]

    self.shape = self.get_image_size()
    self.extract_information()

    if self.information['needs_color_table']:
      self.read_color_table()

    self.read_pixels()
    self.permute_channels(channel = 'RGB')

  def little_to_int(self, bytes):
    return int.from_bytes(bytes, "little")

  def read_image(self, filepath):
    bmp_file = open(filepath, 'rb')
    header = bmp_file.read()
    return header

  def get_image_size(self):
    height = self.little_to_int(self.info_header[8:12])
    width = self.little_to_int(self.info_header[4:8])
    return height, width

  def extract_information(self):
    self.information = process_information(self.shape, self.little_to_int, self.header, self.info_header)
    return

  def read_color_table(self):
    assert self.information['needs_color_table'], "Color table not needed!"
    color_table = self.image[54:54 + self.information['color_table_size']]

    color_map = []

    for i in range(len(color_table)):
      if i % 4 == 0:
        rgb = list(color_table[i:i+3])
        rgb.reverse()
        color_map.append(tuple(rgb))

    self.color_map = color_map

  def read_pixels(self, pixels=None):
    if pixels == None:
      self.pixels = list(self.image[self.information['data_offset']:])
    else:
      self.pixels = pixels

    self.raw_img = [[0] * self.information['width'] for _ in range(self.information['height'])]

    if self.information['needs_color_table']:
      for x in range(self.information['height']):
        for y in range(self.information['width']):
          self.raw_img[x][y] = self.pixels[ - ((x + 1) * self.information['width'] - y) ]
      self.img = [[self.color_map[pixel] for pixel in row] for row in self.raw_img]
    else:
      for x in range(self.information['height']):
        for y in range(self.information['width']):
          coord = ((x + 1) * 3 * self.information['width'] - 3 * y)
          self.raw_img[x][y] = (self.pixels[-coord + 2], self.pixels[-coord + 1], self.pixels[-coord])
      self.img = self.raw_img

    return

  def summary(self):
    for key, value in self.information.items():
      print(f"{key}: {value}")

  def permute_channels(self, channel='RGB', zero=None):
    self.channel = channel
    
    new_r = list(self.channel).index('R')
    new_g = list(self.channel).index('G')
    new_b = list(self.channel).index('B')
    if zero:
        zero_channel = ['R', 'G', 'B'].index(zero)

    for x in range(self.information['height']):
      for y in range(self.information['width']):
        rgb = self.img[x][y]
        new_rgb = [None]*3
        new_rgb[new_r] = rgb[0]
        new_rgb[new_g] = rgb[1]
        new_rgb[new_b] = rgb[2]
        if zero:
            new_rgb[zero_channel] = 0
        self.img[x][y] = new_rgb

  def __repr__(self):
    if hasattr(self, 'shape'):
      return f"{self.__class__.__name__}(\n  filepath='{self.filepath}',\n  shape={self.shape}\n)"
    else:
      return f"{self.__class__.__name__}()"

  def plot(self):
    plt.imshow(self.img)
    plt.axis('off')
    plt.show()
    return

  def write(self, output_path, pixel_array, bits_per_pixel = None, color_map=None):
    if not output_path.endswith('.bmp'):
        raise ValueError("Output file must have a .bmp extension!")
    if isinstance(pixel_array, np.ndarray):
       pixel_array = pixel_array.tolist()

    if type(pixel_array[0][0]) == list:
        bits_per_pixel = 24
    else:
        bits_per_pixel = 8

    self.information = {}
    self.information['width'] = len(pixel_array[0])
    self.information['height'] = len(pixel_array)
    self.information['bits_per_pixel'] = bits_per_pixel

    if bits_per_pixel == 24:
        row_size = (self.information['width'] * 3 + 3) & ~3 
        padding_size = row_size - self.information['width'] * 3
    elif bits_per_pixel == 8:
        row_size = (self.information['width'] + 3) & ~3
        padding_size = row_size - self.information['width']

    pixel_data = bytearray()
    for row in pixel_array[::-1]:
        for pixel in row:
            if bits_per_pixel == 24:
                pixel_data.extend(pixel[::-1])
            elif bits_per_pixel == 8:
                pixel_data.append(pixel)
        pixel_data.extend(b'\x00' * padding_size)

    color_table = bytearray()
    if bits_per_pixel == 8 and color_map:
        for color in color_map:
            r, g, b = color
            color_table.extend([b, g, r, 0x00])
    
    file_size = 14 + 40 + len(color_table) + len(pixel_data)

    header_bytes = b'BM'
    header_bytes += self.int_to_little(file_size, 4)
    header_bytes += b'\x00\x00\x00\x00'  
    header_bytes += self.int_to_little(14 + 40 + len(color_table), 4) 

    info_header_bytes = self.int_to_little(40, 4)  
    info_header_bytes += self.int_to_little(self.information['width'], 4)
    info_header_bytes += self.int_to_little(self.information['height'], 4)
    info_header_bytes += b'\x01\x00'
    info_header_bytes += self.int_to_little(self.information['bits_per_pixel'], 2)
    info_header_bytes += b'\x00\x00\x00\x00'  
    info_header_bytes += self.int_to_little(len(pixel_data), 4)
    info_header_bytes += b'\x13\x0B\x00\x00'  
    info_header_bytes += b'\x13\x0B\x00\x00'
    info_header_bytes += self.int_to_little(len(color_map) if color_map else 0, 4)
    info_header_bytes += b'\x00\x00\x00\x00'

    with open(output_path, 'wb') as bmp_file:
        bmp_file.write(header_bytes)
        bmp_file.write(info_header_bytes)
        bmp_file.write(color_table)
        bmp_file.write(pixel_data)

  def int_to_little(self, value, length):
    return value.to_bytes(length, byteorder='little')

def convolve(image, kernel):
    height = len(image)
    width = len(image[0])
    depth = len(image[0][0])
    k_size = len(kernel)
    pad = k_size // 2
    
    output = [[[0 for _ in range(depth)] for _ in range(width)] for _ in range(height)]
    
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            for d in range(depth):
                conv_sum = 0
    
                for ki in range(k_size):
                    for kj in range(k_size):
                        pixel = image[i + ki - pad][j + kj - pad][d]
                        k_value = kernel[ki][kj]
                        conv_sum += pixel * k_value
    
                output[i][j][d] = min(max(0, conv_sum), 255)
    
    return output

def show_modified(original, modified):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(modified)
    ax[1].set_title('Modified Image')
    ax[1].axis('off')
    plt.show()



def process_information(shape, little_to_int, header, info_header):
    information = {}
    information['width'] = shape[1]
    information['height'] = shape[0]

    information['data_offset'] = little_to_int(header[10:14])
    information['bits_per_pixel'] = little_to_int(info_header[14:16])

    colour_depth = {1: 'Black and White', 8: 'Grayscale', 24: 'True Colour'}
    information['colour_depth'] = colour_depth[information['bits_per_pixel']]

    information['image_size'] = little_to_int(info_header[20:24])
    information['colors_used'] = little_to_int(info_header[32:36])
    information['colors_important'] = little_to_int(info_header[36:40])
    information['needs_color_table'] = information['bits_per_pixel'] <= 8

    if information['needs_color_table']:
      information['color_table_size'] = 4 * information['colors_used']
    else:
      information['color_table_size'] = 0

    information['file_size_kb'] = information['image_size'] / 1000

    return information