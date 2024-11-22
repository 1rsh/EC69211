import numpy as np
from ..bmp import BMP

class ZigZag:
    def __init__(self, block_size=8, zigzag=True):
        self.block_size = block_size

        self.zigzag_index = self._generate_zigzag_index(block_size) if zigzag else self._generate_plain_index(block_size)
    
    def _generate_plain_index(self, n):
        return np.arange(n * n).reshape((n, n))

    def _generate_zigzag_index(self, n):
        index = np.empty((n, n), dtype=int)
        i = 0
        for s in range(2 * n - 1):
            if s % 2 == 0:
                for x in range(max(0, s - n + 1), min(s + 1, n)):
                    index[x, s - x] = i
                    i += 1
            else:
                for x in range(max(0, s - n + 1), min(s + 1, n)):
                    index[s - x, x] = i
                    i += 1
        return index.T

    def zigzag_2d(self, block):
        if block.shape != (self.block_size, self.block_size):
            raise ValueError(f"Block size must be {self.block_size}x{self.block_size}")

        zigzag_ordered_array = np.empty(self.block_size * self.block_size, dtype=block.dtype)
        
        for i in range(self.block_size):
            for j in range(self.block_size):
                zigzag_ordered_array[self.zigzag_index[i, j]] = block[i, j]
        
        return zigzag_ordered_array

    def zigzag_3d(self, blocks):
        height, width, depth = blocks.shape
        if height != self.block_size or width != self.block_size:
            raise ValueError(f"Each block must be {self.block_size}x{self.block_size}")

        result = np.zeros((depth, self.block_size * self.block_size), dtype=blocks.dtype)
        for i in range(depth):
            result[i] = self.zigzag_2d(blocks[..., i])
        return result

    def inverse_zigzag_2d(self, zigzag_array):
        if zigzag_array.size != self.block_size * self.block_size:
            raise ValueError(f"Input array must have size {self.block_size * self.block_size}")

        return zigzag_array[self.zigzag_index.flatten()].reshape(self.block_size, self.block_size)

    def inverse_zigzag_3d(self, zigzag_blocks):
        depth, zigzag_size = zigzag_blocks.shape
        if zigzag_size != self.block_size * self.block_size:
            raise ValueError(f"Zigzag size must be {self.block_size * self.block_size}")

        result = np.zeros((self.block_size, self.block_size, depth), dtype=zigzag_blocks.dtype)
        for i in range(depth):
            result[..., i] = self.inverse_zigzag_2d(zigzag_blocks[i])
        return result
    
if __name__ == "__main__":
    bmp = BMP()
    bmp.read('block.bmp')

    x = np.array(bmp.img)[:8,:8]

    zz = ZigZag(zigzag=True)
    print("Original:\n", x[..., 0])
    z = zz.zigzag_3d(x)
    print("Zigzag:\n", z[0])
    recovered = zz.inverse_zigzag_3d(z)
    print("Recovered: \n", recovered[..., 0])
    print("Check:", np.all(recovered==x))
