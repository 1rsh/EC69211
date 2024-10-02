import numpy as np
from PIL import Image

class Histogram:
    def __init__(self, filepath, T=None) -> None:
        self.image = np.array(Image.open(filepath).convert('RGB'))

        self.hsv_image = self.rgb2hsv(self.image)

        values = self.hsv_image[..., 2]
        self.results = np.stack((self.hsv_image[..., 0], self.hsv_image[..., 1], self.histogram_equalization(values)))
        
    def equalize(self):
        return self.results
    
    def rgb2hsv(self, image):
        image = image.astype('float') / 255.0
        R, G, B = image[..., 0], image[..., 1], image[..., 2]
        
        Cmax = np.max(image, axis=-1)
        Cmin = np.min(image, axis=-1)
        delta = Cmax - Cmin

        H = np.zeros_like(Cmax)
        S = np.zeros_like(Cmax)
        V = Cmax

        mask = delta != 0
        Rmax = (Cmax == R) & mask
        Gmax = (Cmax == G) & mask
        Bmax = (Cmax == B) & mask

        H[Rmax] = (60 * ((G[Rmax] - B[Rmax]) / delta[Rmax]) + 360) % 360
        H[Gmax] = (60 * ((B[Gmax] - R[Gmax]) / delta[Gmax]) + 120) % 360
        H[Bmax] = (60 * ((R[Bmax] - G[Bmax]) / delta[Bmax]) + 240) % 360

        S[Cmax != 0] = delta[Cmax != 0] / Cmax[Cmax != 0]

        hsv_image = np.stack((H, S, V), axis=-1)
        
        return hsv_image
    
    def hsv2rgb(self, hsv_image):
        H, S, V = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]
        
        R = np.zeros_like(H)
        G = np.zeros_like(H)
        B = np.zeros_like(H)
        
        Hi = (H // 60).astype(int) % 6
        
        f = (H / 60) - Hi
        p = V * (1 - S)
        q = V * (1 - f * S)
        t = V * (1 - (1 - f) * S)
        
        R[Hi == 0] = V[Hi == 0]
        G[Hi == 0] = t[Hi == 0]
        B[Hi == 0] = p[Hi == 0]
        
        R[Hi == 1] = q[Hi == 1]
        G[Hi == 1] = V[Hi == 1]
        B[Hi == 1] = p[Hi == 1]
        
        R[Hi == 2] = p[Hi == 2]
        G[Hi == 2] = V[Hi == 2]
        B[Hi == 2] = t[Hi == 2]
        
        R[Hi == 3] = p[Hi == 3]
        G[Hi == 3] = q[Hi == 3]
        B[Hi == 3] = V[Hi == 3]
        
        R[Hi == 4] = t[Hi == 4]
        G[Hi == 4] = p[Hi == 4]
        B[Hi == 4] = V[Hi == 4]
        
        R[Hi == 5] = V[Hi == 5]
        G[Hi == 5] = p[Hi == 5]
        B[Hi == 5] = q[Hi == 5]
        
        rgb_image = np.stack((R, G, B), axis=-1)
        
        rgb_image = (rgb_image * 255).astype('uint8')
        
        return rgb_image

    def calculate_histogram(self, image):
        histogram = np.zeros(256)
        for pixel in image.ravel():
            histogram[pixel] += 1
        return histogram

    def calculate_cdf(self, histogram):
        cdf = histogram.cumsum()
        cdf_normalized = cdf / cdf[-1]
        return cdf_normalized

    def get_cdf(self, image):
        if len(image.shape) == 2:
            histogram = self.calculate_histogram(image)
            cdf = self.calculate_cdf(histogram)
            return cdf
        else:
            cdfs = []
            for img in [image[:,:,0], image[:,:,1], image[:,:,2]]:
                histogram = self.calculate_histogram(img)
                cdf = self.calculate_cdf(histogram)
                cdfs.append(cdf)
            return cdfs

    def histogram_equalization(self, image, T=None):
        if T is None:
            self.histogram = self.calculate_histogram(image)
            self.cdf = self.calculate_cdf(self.histogram)
            T = np.floor(255 * self.cdf).astype(np.uint8)

        equalized_image = T[image]
        return equalized_image

    def histogram_matching(self, target_image):
        return self.match_images(self.image, target_image)
    
    def match_images(self, source, target):
        hsv_image = self.rgb2hsv(source)
        source_vals = (hsv_image[..., 2] * 255).astype(int)
        target_vals = (self.rgb2hsv(target)[..., 2] * 255).astype(int)

        source_histogram = self.calculate_histogram(source_vals)
        target_histogram = self.calculate_histogram(target_vals)

        source_cdf = self.calculate_cdf(source_histogram)
        target_cdf = self.calculate_cdf(target_histogram)

        T = np.zeros(256, dtype=np.uint8)
        target_value = 0
        for source_value in range(256):
            while target_value < 255 and target_cdf[target_value] < source_cdf[source_value]:
                target_value += 1
            T[source_value] = target_value

        matched_values = self.histogram_equalization(source_vals, T=T)

        modified_hsv = np.moveaxis(np.stack((hsv_image[..., 0], hsv_image[..., 1], matched_values / 255.0)), 0, 2)

        return self.hsv2rgb(modified_hsv)