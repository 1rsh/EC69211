import numpy as np
import matplotlib.pyplot as plt

from .lumchrom import ColorConverter
from .subsample import Subsampler
from .block import BlockProcessor
from ..bmp.image import BMP

class Quantizer:
    def __init__(self, quality: int = 50):
        # Default JPEG quantization tables (standardized)
        self.luminance_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

        self.chrominance_table = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ])

        # Adjust quantization tables based on quality factor
        self.luminance_table = self.adjust_quality(self.luminance_table, quality)
        self.chrominance_table = self.adjust_quality(self.chrominance_table, quality)

    def get_quantization_table(self, component='Y'):
        if component == 'Y':
            return self.luminance_table
        else:
            return self.chrominance_table

    def adjust_quality(self, table: np.ndarray, quality: int) -> np.ndarray:
        """
        Adjust the quantization table based on the quality factor.
        Higher quality -> lower compression.
        """
        if quality < 1:
            quality = 1
        elif quality > 100:
            quality = 100

        # Scale quality factor
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - quality * 2

        adjusted_table = np.floor((table * scale + 50) / 100)
        adjusted_table[adjusted_table < 1] = 1
        adjusted_table[adjusted_table > 255] = 255
        return adjusted_table

    def quantize(self, block: np.ndarray) -> np.ndarray:
        """
        Quantize the given 8x8 block using the appropriate quantization table.
        Component can be 'luminance' or 'chrominance'.
        """
        for channel in range(3):
            if channel == 0:
                block[channel] = np.round(block[channel] / self.luminance_table).astype(np.int32)
            else:
                block[channel] = np.round(block[channel] / self.chrominance_table).astype(np.int32)
        return block
        # if block.ndim == 3:
        #     y, cb, cr = (block[..., i] for i in range(3))
        #     return np.stack([
        #         np.round(y / self.luminance_table).astype(np.int32),
        #         np.round(cb / self.chrominance_table).astype(np.int32),
        #         np.round(cr / self.chrominance_table).astype(np.int32)
        #     ], axis=-1)
        # else:
        #     return np.stack([self.quantize(b) for b in block], axis=0)
        # else:
        #     raise ValueError("Input should have 3 or 4 dims")

    def dequantize(self, block: np.ndarray, component: str = 'luminance') -> np.ndarray:
        """
        Dequantize the given 8x8 block using the appropriate quantization table.
        Component can be 'luminance' or 'chrominance'.
        """
        for channel in range(3):
            if channel == 0:
                block[channel] = (block[channel] * self.luminance_table).astype(np.int32)
            else:
                block[channel] = (block[channel] * self.chrominance_table).astype(np.int32)
        return block
        # if block.ndim == 3:
        #     y, cb, cr = (block[..., i] for i in range(3))
        #     return np.stack([
        #         np.round(y * self.luminance_table).astype(np.int32),
        #         np.round(cb * self.chrominance_table).astype(np.int32),
        #         np.round(cr * self.chrominance_table).astype(np.int32)
        #     ], axis=-1)
        # elif block.ndim == 4:
        #     return np.stack([self.dequantize(b) for b in block], axis=0)
        # else:
        #     raise ValueError("Input should have 3 or 4 dims")


if __name__ == "__main__":
    # Assuming BMP class handles reading and writing BMP files correctly
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
    print(np.max(dct_blocks[0]))
    
    q = Quantizer(quality=95)
    qdct_blocks = q.quantize(dct_blocks)

    print(np.max(qdct_blocks[0]))

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
