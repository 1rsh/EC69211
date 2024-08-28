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