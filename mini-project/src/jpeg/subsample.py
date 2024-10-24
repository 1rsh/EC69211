import numpy as np
import matplotlib.pyplot as plt

from .lumchrom import ColorConverter
from ..bmp.image import BMP

class Subsampler:
    def __init__(self, method='4:2:0'):
        self.method = method

    def _split_dims(self, image, upsample=False):
        if upsample:
            Y, Cb, Cr = (image[i] for i in range(3))
        else:
            Y, Cb, Cr = (image[..., i] for i in range(3))
        return Y, Cb, Cr
    
    def _join_dims(self, Y, Cb, Cr, upsample=False):
        if upsample:
            return np.stack((Y, Cb, Cr), axis=-1)
        img = np.empty(3, dtype=object)
        img[0] = Y
        img[1] = Cb
        img[2] = Cr
        return img
    
    def _odd_append(self, sub_image):
        if sub_image.shape[0] % 2:
            sub_image = np.vstack([sub_image, sub_image[-1, :]])
        if sub_image.shape[1] % 2:
            sub_image = np.vstack([sub_image, sub_image[:, -1]])
        return sub_image

    def subsample(self, image):
        """
        Subsample Cb and Cr according to the method.
        """
        Y, Cb, Cr = self._split_dims(image)

        if self.method == '4:4:4':
            return self._join_dims(Y, Cb, Cr)
        
        elif self.method == '4:2:2':
            Cb_sub = self._odd_append(Cb[:, ::2])
            Cr_sub = self._odd_append(Cr[:, ::2])
            
            return self._join_dims(Y, Cb_sub, Cr_sub)

        elif self.method == '4:2:0':
            Cb_sub = self._odd_append(Cb[::2, ::2])
            Cr_sub = self._odd_append(Cr[::2, ::2])

            return self._join_dims(Y, Cb_sub, Cr_sub)
        
        else:
            raise ValueError(f"Unsupported subsampling method: {self.method}")


    def upsample(self, image):
        """
        Upsample Cb and Cr to match Y's resolution.
        """
        Y, Cb, Cr = self._split_dims(image, upsample=True)

        if self.method == '4:4:4':
            return self._join_dims(Y, Cb, Cr, upsample=True)
        
        elif self.method == '4:2:2':
            Cb_up = np.repeat(Cb, 2, axis=1)
            Cr_up = np.repeat(Cr, 2, axis=1)
            return self._join_dims(Y, Cb_up, Cr_up, upsample=True)
        
        elif self.method == '4:2:0':
            Cb_up = np.repeat(np.repeat(Cb, 2, axis=0), 2, axis=1)
            Cr_up = np.repeat(np.repeat(Cr, 2, axis=0), 2, axis=1)
            return self._join_dims(Y, Cb_up, Cr_up, upsample=True)
        
        else:
            raise ValueError(f"Unsupported upsampling method: {self.method}")
        
if __name__ == "__main__":
    bmp = BMP()
    bmp.read('data/iitkgp.bmp')

    cc = ColorConverter()
    ycc_img = cc.rgb2ycbcr(bmp.img)
    
    ss = Subsampler(method='4:2:2')
    subsampled = ss.subsample(ycc_img)
    
    upsampled = ss.upsample(subsampled)
    reconstructed = cc.ycbcr2rgb(upsampled)

    plt.subplot(1, 2, 1)
    plt.imshow(bmp.img)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed)
    plt.axis('off')
    plt.show()


