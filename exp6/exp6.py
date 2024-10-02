import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from histogram_utils import Histogram
import os

class LabImage(Histogram):
    def __init__(self, filepath, T=None) -> None:
        self.image = np.array(Image.open(filepath).convert('RGB'))
        self.hsv_image = self.rgb2hsv(self.image)

        self.values = self.get_values(self.image)
        self.modified_values = self.histogram_equalization(self.values)

        modified_hsv = np.moveaxis(np.stack((self.hsv_image[..., 0], self.hsv_image[..., 1], self.modified_values / 255.0)), 0, 2)
        self.results = self.hsv2rgb(modified_hsv)

    def get_values(self, image):
        return (self.rgb2hsv(image)[..., 2] * 255).astype(int)

    def plot_equalize(self):
        plt.figure(figsize=(12, 6))
        plt.suptitle("Histogram Equalization (Values)")

        plt.subplot(2, 3, 1)
        plt.axis('off')
        plt.imshow(self.image, cmap="gray")

        plt.subplot(2, 3, 2)
        cdf = np.array(self.get_cdf(self.values))
        plt.plot(cdf)

        plt.subplot(2, 3, 3)
        plt.yticks([])
        pdf = np.diff(cdf, prepend=0)
        plt.plot(range(256), pdf)
        plt.fill_between(range(256), pdf, alpha=0.3)

        plt.subplot(2, 3, 4)
        plt.axis('off')
        plt.imshow(self.results, cmap="gray")
        plt.subplot(2, 3, 5)
        cdf = np.array(self.get_cdf(self.modified_values))
        plt.plot(cdf)
        plt.subplot(2, 3, 6)
        plt.yticks([])
        pdf = np.diff(cdf, prepend=0)
        plt.plot(range(256), pdf)
        plt.fill_between(range(256), pdf, alpha=0.3)
        plt.show()

    def plot_matched(self, target_image=None):
        matched_image = self.equalize() if target_image is None else self.histogram_matching(target_image)

        plt.suptitle("Histogram Matching")

        plt.subplot(2, 2, 1)
        plt.axis('off')
        plt.imshow(self.image, cmap="gray")
        plt.title("Source Image")
        
        plt.subplot(2, 2, 2)
        plt.axis('off')
        plt.imshow(target_image, cmap="gray")
        plt.title("Target Image")
        
        plt.subplot(2, 2, 3)
        plt.axis('off')
        plt.imshow(matched_image, cmap="gray")
        plt.title("Matched Image")
        
        plt.subplot(2, 2, 4)
        plt.plot(self.get_cdf(self.get_values(self.image)), label="Source CDF")
        plt.plot(self.get_cdf(self.get_values(target_image)), label="Target CDF", linestyle='dashed')
        plt.plot(self.get_cdf(self.get_values(matched_image)), label="Matched CDF", linestyle='dotted')
        plt.legend()
        plt.show()
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str, help="Source image filepath")
    parser.add_argument("-t", type=str, help="Target image filepath", required=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print("\033[92m"+"="*os.get_terminal_size().columns)
    print(f"\nImage and Video Processing Laboratory | EC69211 | Experiment - 6")
    print("\nCode Author: Irsh Vijay | 21EC39055\n")
    print("-"*os.get_terminal_size().columns + "\n\033[0m")
    source_image = LabImage(args.s)

    if args.t:
        print(f"Histogram Matching:\nSource: {args.s} -> Target: {args.t}\n")
        target_image = LabImage(args.t).image
        source_image.plot_matched(target_image)
    else:
        print(f"Histogram Equalization:\nSource: {args.s}\n")
        source_image.plot_equalize()
    
    print("\033[92m" + "-"*os.get_terminal_size().columns + "\n\033[0m")