import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .block import BlockProcessor
from .encode import EntropyEncoder
from .lumchrom import ColorConverter
from .quantize import Quantizer
from .subsample import Subsampler
from .zigzag import ZigZag
from .huffman import huff, dehuff
from .save import save_to_file, load_from_file
from ..bmp import BMP

class ImageCompressor:
    def __init__(self, quality_factor=50, subsampling='4:2:0', verbose=False):
        self.quality_factor = quality_factor
        self.subsampling = subsampling

        self.compressed_data = None
        self.decompressed_image = None

        self.cc = ColorConverter()
        self.ss = Subsampler(subsampling)
        self.bp = BlockProcessor(verbose=verbose)
        self.q = Quantizer(self.quality_factor)
        self.zz = ZigZag()
        self.ee = EntropyEncoder()

    def compress(self, image):
        """Orchestrate the compression pipeline."""
        # Step 1: Convert RGB to YCbCr
        self.original_image = image
        self.shape = self.original_image.shape
        ycbcr_image = self.cc.rgb2ycbcr(self.original_image)

        # Step 2: Apply chroma subsampling (optional)
        subsampled_image = self.ss.subsample(ycbcr_image)
        
        # Step 3: Split into blocks
        blocks = self.bp.split_into_blocks(subsampled_image)

        # Step 4: Apply DCT
        dct_blocks = self.bp.apply_dct(blocks)

        # Step 5: Quantization
        quantized_blocks = self.q.quantize(dct_blocks)

        # Step 6: Zigzag scan and RLE (optional)
        compressed_data = []
        for i in range(3):
            zigzagged_blocks = np.stack([self.zz.zigzag_2d(qb) for qb in quantized_blocks[i]]).astype(int)
            compressed = self.ee.compress_blocks(zigzagged_blocks)
            compressed_data.append(huff(compressed))

        return compressed_data

    def decompress(self, data):
        """Orchestrate the decompression pipeline."""
        compressed_data = []
        for channel in range(3):
            compressed_data.append(dehuff(*data[channel]))
            
        # compressed_data = self.byte_string_to_lists(raw_bytes)
        # Step 1: Run-length decode
        quantized_blocks = np.empty(3, dtype=object)
        for i in range(3):
            zigzagged_blocks = self.ee.decompress_blocks(compressed_data[i])
            quantized_blocks[i] = np.stack([self.zz.inverse_zigzag_2d(z) for z in zigzagged_blocks])
        
        # Step 2: Inverse quantization
        dequantized_blocks = self.q.dequantize(quantized_blocks)
        
        # Step 3: Apply inverse DCT
        idct_blocks = self.bp.apply_idct(dequantized_blocks)
        # Step 4: Merge blocks into image

        ycbcr_image = self.bp.merge_blocks(idct_blocks, self.shape, subsampling=self.subsampling)
        
        upsampled_image = self.ss.upsample(ycbcr_image)

        # Step 5: Convert YCbCr back to RGB
        self.decompressed_image = self.cc.ycbcr2rgb(upsampled_image)
    
    def save(self, filename, compressed_data):
        save_to_file(compressed_data, filename)
    
    def load(self, filename):
        return load_from_file(filename)

if __name__ == "__main__":
    bmp = BMP()
    bmp.read("data/iitkgp.bmp")

    ic = ImageCompressor(quality_factor=95)
    raw_bytes = ic.compress(np.array(bmp.img))
    ic.save("try.jpg", raw_bytes)
    ic.decompress(raw_bytes)
    
    plt.subplot(1, 2, 1)
    plt.imshow(bmp.img)
    plt.subplot(1, 2, 2)
    plt.imshow(ic.decompressed_image)
    plt.show()