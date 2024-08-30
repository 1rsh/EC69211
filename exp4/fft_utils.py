from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class FFT:
    def __init__(self) -> None:
        pass

    def fft_1d(self, x):
        """
        Recursive FFT function
        """
        n = len(x)

        if not (n & (n - 1)) == 0:
            next_pow_2 = 1 << (n - 1).bit_length()
            x = np.concatenate([x, np.zeros(next_pow_2 - n)])
            n = next_pow_2

        if n <= 1:
            return x

        x_even = self.fft_1d(x[::2])
        x_odd = self.fft_1d(x[1::2])

        muls = np.exp(-2j * np.pi * np.arange(n//2) / n) * x_odd

        ans = np.concatenate([
            x_even + muls,
            x_even - muls
        ])

        return ans

    def ifft_1d(self, x):
        """
        IFFT function
        """
        n = len(x)
        x_bar = [y.conjugate() for y in x]
        fft = self.fft_1d(x_bar)
        ifft = [y.conjugate() / n for y in fft]
        return ifft

    def fft_2d(self, image):
        """
        FFT function for Image
        1. FFT for rows is calculated
        2. FFT of the previous step on columns is calculated
        """
        rows, cols = image.shape

        fft_rows = []
        for r in range(rows):
            fft_rows.append(self.fft_1d(image[r]))

        fft_rows = np.array(fft_rows)

        fft_cols = []
        for c in range(cols):
            fft_cols.append(self.fft_1d(fft_rows[:, c]))

        res = np.array(fft_cols)

        return res

    def ifft_2d(self, fft):
        """
        IFFT function for Image
        1. IFFT for rows is calculated
        2. IFFT of the previous step on columns is calculated
        """
        rows, cols = fft.shape

        ifft_rows = []
        for r in range(rows):
            ifft_rows.append(self.ifft_1d(fft[r]))

        ifft_rows = np.array(ifft_rows)

        ifft_cols = []
        for c in range(cols):
            ifft_cols.append(self.ifft_1d(ifft_rows[:, c]))

        res = np.array(ifft_cols)

        return res

    def visualize_spectrum(self, fft, shift=True, title=None):
        if shift:
            fft = self.fftshift(fft)

        mag = np.abs(fft)
        phase = np.angle(fft)

        plt.figure()
        if title:
            plt.title(title)
            plt.axis('off')
        plt.subplot(1, 2, 1)
        plt.title('Magnitude Spectrum (log)')
        plt.imshow(np.log(mag + 1), cmap='gist_heat')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Phase Spectrum')
        plt.imshow(phase, cmap='gist_heat')
        plt.axis('off')

        plt.tight_layout()

        return

    def load_image(self, filepath):
        self.filepath = filepath
        img = Image.open(filepath).convert('L')
        self.img = np.array(img)

    def fftshift(self, x):
        """
        Shift zero-frequency component to the center of the spectrum
        """
        rows, cols = x.shape
        row_mid = rows // 2
        col_mid = cols // 2

        shifted = np.zeros_like(x, dtype=x.dtype)

        shifted[:row_mid, :col_mid] = x[row_mid:, col_mid:]
        shifted[row_mid:, col_mid:] = x[:row_mid, :col_mid]
        shifted[:row_mid, col_mid:] = x[row_mid:, :col_mid]
        shifted[row_mid:, :col_mid] = x[:row_mid, col_mid:]

        return shifted