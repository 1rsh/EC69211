import numpy as np
import matplotlib.pyplot as plt

from .block import BlockProcessor
from .zigzag import ZigZag
from .lumchrom import ColorConverter
from .subsample import Subsampler
from .quantize import Quantizer
from .huffman import huff, dehuff
from .save import save_to_file, load_from_file
from ..bmp import BMP

class EntropyEncoder:
    def __init__(self):
        self.encoded_data = []
    
    def run_length_encode(self, flat_block):
        # Flatten the block for easier processing
        self.encoded_data = []
        
        count = 0
        for value in flat_block:
            if value == 0:
                count += 1
            else:
                if count > 0:
                    self.encoded_data.extend((0, count))  # Store trailing zeros
                    count = 0
                self.encoded_data.extend((value, 1))  # Store value and its count
        
        if count > 0:
            self.encoded_data.extend((0, count))  # Handle trailing zeros at the end
        
        if self.encoded_data[-2] == 0:
            self.encoded_data[-1] = 0
        else:
            self.encoded_data.extend((0, 0))
        return self.encoded_data
    
    def compress_blocks(self, flat_blocks):
        encoded = []
        prev_dc = 0
        for b in flat_blocks:
            dc = b[0]
            encoded.append(dc - prev_dc)
            prev_dc = dc
            e = self.run_length_encode(b[1:].tolist())
            encoded.append(e)

        return encoded
    
    def decompress_blocks(self, encoded):
        decompressed_blocks = []
        prev_dc = 0
        index = 0

        while index < len(encoded):
            dc_diff = encoded[index]  # DC differential value
            dc_value = dc_diff + prev_dc  # Restore the actual DC value
            prev_dc = dc_value  # Update prev_dc for the next block

            index += 1

            ac_encoded = encoded[index]
            ac_decoded = self.run_length_decode(ac_encoded)

            full_block = np.zeros(64, dtype=int) 
            full_block[0] = dc_value  
            full_block[1:] = ac_decoded

            decompressed_blocks.append(full_block)

            index += 1

        return np.array(decompressed_blocks)
    
    def run_length_decode(self, encoded_data):
        flat_block = np.zeros(63, dtype=int)
        index = 0
        
        for i in range(0, len(encoded_data), 2):
            value = encoded_data[i]
            count = encoded_data[i+1]
            if value == 0:
                index += count 
            else:
                for _ in range(count):
                    flat_block[index] = value
                    index += 1
                    
        return flat_block
    
def format_compressed(compressed):
    _compressed = []
    for c in compressed:
        if isinstance(c, list):
            _compressed.extend(zip(c[::2], c[1::2]))
        else:
            _compressed.append(c)
    return _compressed

if __name__ == "__main__":
    bmp = BMP()
    bmp.read('data/input.bmp')

    cc = ColorConverter()
    ycc_img = cc.rgb2ycbcr(bmp.img)
    
    subsampling = '4:2:0'
    ss = Subsampler(method=subsampling)
    subsampled = ss.subsample(ycc_img)

    bp = BlockProcessor(block_size=8)
    blocks = bp.split_into_blocks(subsampled, subsampling=subsampling)

    dct_blocks = bp.apply_dct(blocks)
    
    q = Quantizer(quality=100)
    qdct_blocks = q.quantize(dct_blocks)

    zz = ZigZag()
    ee = EntropyEncoder()

    compressed_data = []
    for i in range(3):
        zigzagged_blocks = np.stack([zz.zigzag_2d(qb) for qb in qdct_blocks[i]]).astype(int)
        compressed = ee.compress_blocks(zigzagged_blocks)
        compressed_data.append(huff(compressed))
    
    save_to_file(compressed_data, "data/output.myjpg")

    new_compressed_data = load_from_file("data/output.myjpg")
    print(compressed_data == new_compressed_data)

    # decoding
    qdct_blocks = np.empty(3, dtype=object)
    for i in range(3):
        r_zigzag = ee.decompress_blocks(dehuff(*compressed_data[i]))
        qdct_blocks[i] = np.stack([zz.inverse_zigzag_2d(z) for z in r_zigzag])

    dct_blocks = q.dequantize(qdct_blocks)

    il = bp.apply_idct(dct_blocks)

    print("Check:", [np.allclose(il[i], blocks[i], atol=1) for i in range(3)])
    
    merged = bp.merge_blocks(il, image_shape=np.array(bmp.img).shape, subsampling=subsampling)

    print("Intermediate Shape:", [merged[i].shape for i in range(3)])

    upsampled = ss.upsample(merged)
    reconstructed = cc.ycbcr2rgb(upsampled)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(bmp.img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed)
    plt.title("IDCT of DCT")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
