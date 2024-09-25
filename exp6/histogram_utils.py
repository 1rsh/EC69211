import numpy as np
from PIL import Image

class Histogram:
    def __init__(self, filepath, T=None) -> None:
        self.image = np.array(Image.open(filepath))
        if len(self.image.shape) == 2:
            self.results = self.histogram_equalization(self.image, T=T)
        else:
            res = []
            for img in [self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]]:
                res.append(self.histogram_equalization(img, T=T))
            self.results = np.moveaxis(np.stack(res), 0, 2)

    def equalize(self):
        return self.results

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
        if len(self.image.shape) == 2:
            return self.match_single_channel(self.image, target_image)
        else:
            matched_channels = []
            for i in range(3):
                matched_channel = self.match_single_channel(self.image[:, :, i], target_image[:, :, i])
                matched_channels.append(matched_channel)
            return np.stack(matched_channels, axis=-1)

    def match_single_channel(self, source, target):
        source_histogram = self.calculate_histogram(source)
        target_histogram = self.calculate_histogram(target)

        source_cdf = self.calculate_cdf(source_histogram)
        target_cdf = self.calculate_cdf(target_histogram)

        T = np.zeros(256, dtype=np.uint8)
        target_value = 0
        for source_value in range(256):
            while target_value < 255 and target_cdf[target_value] < source_cdf[source_value]:
                target_value += 1
            T[source_value] = target_value

        matched_image = self.histogram_equalization(source, T=T)
        return matched_image