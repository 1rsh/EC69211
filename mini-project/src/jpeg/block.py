import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocess as mp
import numba as nb

from ..bmp import BMP
from .lumchrom import ColorConverter
from .subsample import Subsampler

class FastCosineTransform:
    def __init__(self):
        pass
    
    @staticmethod
    @nb.jit(nopython=True)
    def dct1d(a):
        N = len(a)
        result = np.zeros_like(a, dtype=np.float64)
        factor = np.pi / (2 * N)

        for k in range(N):
            sum_val = 0.0
            for n in range(N):
                sum_val += a[n] * np.cos((2 * n + 1) * k * factor)
            if k == 0:
                result[k] = sum_val * np.sqrt(1 / N)
            else:
                result[k] = sum_val * np.sqrt(2 / N)

        return result

    @staticmethod
    @nb.jit(nopython=True)
    def idct1d(a):
        N = len(a)
        result = np.zeros_like(a, dtype=np.float64)
        factor = np.pi / (2 * N)

        for n in range(N):
            sum_val = a[0] * np.sqrt(1 / N)
            for k in range(1, N):
                sum_val += a[k] * np.sqrt(2 / N) * np.cos((2 * n + 1) * k * factor)
            result[n] = sum_val

        return result

    def dct_2d(self, a):
        # Apply 1D DCT on rows
        a_dct = np.apply_along_axis(self.dct1d, axis=0, arr=a)
        # Apply 1D DCT on columns
        a_dct = np.apply_along_axis(self.dct1d, axis=1, arr=a_dct)
        return a_dct

    def idct_2d(self, a):
        # Apply 1D IDCT on rows
        a_idct = np.apply_along_axis(self.idct1d, axis=0, arr=a)
        # Apply 1D IDCT on columns
        a_idct = np.apply_along_axis(self.idct1d, axis=1, arr=a_idct)
        return a_idct

class BlockProcessor:
    def __init__(self, block_size=8, verbose=False):
        self.block_size = block_size
        self.dct = FastCosineTransform()
        self.num_workers = mp.cpu_count() - 1
        self.verbose=verbose
    
    def _pad_image(self, image):
        new_image = np.empty(3, dtype=object)
        for channel in range(3):
            h, w = image[channel].shape
            h = int(np.ceil(h/self.block_size)*self.block_size)
            w = int(np.ceil(w/self.block_size)*self.block_size)

            new_image[channel] = np.zeros((h, w))
            new_image[channel][:image[channel].shape[0], :image[channel].shape[1]] = image[channel]
        return new_image
        

    def split_into_blocks(self, image, subsampling='4:2:2'):
        """
        Split the image into non-overlapping blocks of size block_size x block_size.

        Args:
            image (numpy.ndarray): Input image as a 2D or 3D numpy array.

        Returns:
            List[numpy.ndarray]: List of image blocks.
        """
        image = self._pad_image(image)

        blocks = []
        for channel in range(len(image)):  # Iterate over color channels
            height, width = self.block_size, self.block_size
            if subsampling == "4:2:2" and channel > 0:
                width //= 2
            elif subsampling == "4:2:0" and channel > 0:
                height //= 2
                width //= 2
            channel_blocks = self._split_channel_into_blocks(image[channel], (self.block_size, self.block_size))
            blocks.append(channel_blocks)
        
        N = len(channel_blocks)

        npblocks = np.empty(3, dtype=object)
        for i in range(3):
            npblocks[i] = np.array(blocks[i]).astype(int)

        return npblocks

    def _split_channel_into_blocks(self, channel, block_size):
        """
        Helper method to split a single channel image into blocks.

        Args:
            channel (numpy.ndarray): Single channel of the image.

        Returns:
            List[numpy.ndarray]: List of blocks for the channel.
        """
        height, width = channel.shape

        blocks = []
        for i in range(0, height, block_size[0]):
            for j in range(0, width, block_size[1]):
                block = channel[i:i+block_size[0], j:j+block_size[1]]
                if block.shape[0] == block_size[0] and block.shape[1] == block_size[1]:
                    blocks.append(block)

        return blocks
    
    def merge_blocks(self, blocks, image_shape, subsampling="4:2:0"):
        """
        Merge blocks back into the original image shape.

        Args:
            blocks (numpy.ndarray): List of blocks as a numpy array.
            image_shape (tuple): Shape of the original image (height, width) for grayscale
                                or (height, width, channels) for color images.

        Returns:
            numpy.ndarray: Reconstructed image from blocks.
        """
        
        image = np.empty(3, dtype=object)
        for channel in range(3):
            block = blocks[channel]
            height, width, _ = image_shape

            block_size = [self.block_size, self.block_size]
            if subsampling == "4:2:2" and channel > 0:
                width //= 2
                # block_size[1]//=2
            elif subsampling == "4:2:0" and channel > 0:
                height //= 2
                width //= 2
                # block_size[1]//=2
                # block_size[0]//=2
            
            h = int(np.ceil(height/self.block_size)*self.block_size)
            w = int(np.ceil(width/self.block_size)*self.block_size)

            reconstructed_image = np.zeros((h, w), dtype=block.dtype)
            block_idx = 0
            for i in range(0, h, block_size[0]):
                for j in range(0, w, block_size[1]):
                    if block_idx < len(block):
                        reconstructed_image[i:i+block_size[0], j:j+block_size[1]] = block[block_idx]
                    block_idx += 1

            image[channel] = reconstructed_image[:height, :width]
        return image


    def apply_dct(self, blocks):
        """
        Apply Discrete Cosine Transform (DCT) to each block.

        Args:
            blocks (List[numpy.ndarray]): List of image blocks.

        Returns:
            List[numpy.ndarray]: List of DCT-transformed blocks.
        """

        def process_block(block):
            return self.dct.dct_2d(block)
        
        npdct = np.empty(3, dtype=object)

        for channel in range(3):
            with mp.Pool(self.num_workers) as p:
                if self.verbose:
                    blocks[channel] = tqdm(blocks[channel])
                dct_blocks = p.map(process_block, blocks[channel])
                npdct[channel] = np.array(dct_blocks)
        return npdct

    def apply_idct(self, dct_blocks):
        """
        Apply Inverse Discrete Cosine Transform (IDCT) to each DCT block.

        Args:
            dct_blocks (List[numpy.ndarray]): List of DCT blocks.

        Returns:
            List[numpy.ndarray]: List of IDCT-transformed blocks.
        """
        # def process_block(block):
        #     channels = [self.dct.idct_2d(block[..., channel]) for channel in range(dct_blocks.shape[-1])]
        #     return channels
        
        # with mp.Pool(self.num_workers) as p:
        #     idct_blocks = p.map(process_block, tqdm(dct_blocks))
        #     idct_blocks = np.array(idct_blocks)
        #     idct_blocks = np.transpose(np.array(idct_blocks), axes=[0, 2, 3, 1])
        #     idct_blocks = np.clip(idct_blocks, a_min=0, a_max=255).astype(int)
        # return idct_blocks
        def process_block(block):
            return self.dct.idct_2d(block)
        
        npidct = np.empty(3, dtype=object)

        for channel in range(3):
            with mp.Pool(self.num_workers) as p:
                if self.verbose:
                    dct_blocks[channel] = tqdm(dct_blocks[channel])
                idct_blocks = p.map(process_block, dct_blocks[channel])
                npidct[channel] = np.array(idct_blocks).astype(int)
        return npidct
    
    # def __getstate__(self):
    #     self_dict = self.__dict__.copy()
    #     del self_dict['pool']
    #     return self_dict

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    
if __name__ == "__main__":
    # Assuming BMP class handles reading and writing BMP files correctly
    bmp = BMP()
    bmp.read('data/input.bmp')

    cc = ColorConverter()
    ycc_img = cc.rgb2ycbcr(bmp.img)
    
    subsampling = '4:2:0'
    ss = Subsampler(method=subsampling)
    subsampled = ss.subsample(ycc_img)
    print("Subsampled Shape:", [subsampled[i].shape for i in range(3)])

    bp = BlockProcessor(block_size=8)
    blocks = bp.split_into_blocks(subsampled, subsampling=subsampling)
    print("Blocks Shape:", [len(blocks[i]) for i in range(3)])

    dct_blocks = bp.apply_dct(blocks)
    print("DCT Shape:", [len(dct_blocks[i]) for i in range(3)])

    il = bp.apply_idct(dct_blocks)
    print("Check:", [np.allclose(il[i], blocks[i], atol=1) for i in range(3)])
    
    merged = bp.merge_blocks(il, image_shape=np.array(bmp.img).shape, subsampling=subsampling)

    print("Intermediate Shape:", [merged[i].shape for i in range(3)])

    upsampled = ss.upsample(merged)
    print(upsampled.shape)
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
