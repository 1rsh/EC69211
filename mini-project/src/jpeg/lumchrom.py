import numpy as np
import matplotlib.pyplot as plt

from ..bmp.image import BMP

class ColorConverter:
    def __init__(self):
        self.transform_matrix = np.array([[0.299, 0.587, 0.114],
                                    [-0.168736, -0.331264, 0.5],
                                    [0.5, -0.418688, -0.081312]])
        
        self.inv_transform_matrix = np.array([[1.0, 0.0, 1.402],
                                    [1.0, -0.344136, -0.714136],
                                    [1.0, 1.772, 0.0]])
        
    def rgb2ycbcr(self, image: np.ndarray) -> np.ndarray:
        """
        Converts an RGB image to YCbCr color space.
        
        :param image: A NumPy array of shape (height, width, 3) representing the RGB image.
        :return: A NumPy array of shape (height, width, 3) representing the YCbCr image.
        """
        ycbcr_image = image @ self.transform_matrix.T + np.array([0, 128, 128])
        
        return ycbcr_image.astype(np.uint8)

    def ycbcr2rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Converts a YCbCr image back to RGB color space.
        
        :param image: A NumPy array of shape (height, width, 3) representing the YCbCr image.
        :return: A NumPy array of shape (height, width, 3) representing the RGB image.
        """
        temp = image - np.array([0, 128, 128])
        flat = temp.reshape(-1, 3)

        rgb_flat = np.dot(flat, self.inv_transform_matrix.T)
        rgb_image = rgb_flat.reshape(image.shape[0], image.shape[1], 3)

        return np.clip(rgb_image, 0, 255).astype(np.uint8)
    
if __name__ == "__main__":
    bmp = BMP()
    bmp.read('data/input_image.bmp')

    cc = ColorConverter()
    ycc_img = cc.rgb2ycbcr(bmp.img)
    reconstructed = cc.ycbcr2rgb(ycc_img)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.imshow(bmp.img)
    plt.axis('off')
    plt.title("RGB")
    plt.subplot(2, 2, 2)
    plt.imshow(ycc_img[..., 0], cmap="gray")
    plt.axis('off')
    plt.title("Y")
    plt.subplot(2, 2, 3)
    plt.imshow(ycc_img[..., 1], cmap="gray")
    plt.axis('off')
    plt.title("Cb")
    plt.subplot(2, 2, 4)
    plt.imshow(ycc_img[..., 2], cmap="gray")
    plt.axis('off')
    plt.title("Cr")
    plt.show()

