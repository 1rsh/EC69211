import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from histogram_utils import Histogram
import os

class LabImage(Histogram):
    def __init__(self, filepath, T=None) -> None:
        self.image = np.array(Image.open(filepath))
        if len(self.image.shape) == 2:
            self.results = self.histogram_equalization(self.image, T=T)
        else:
            res = []
            for img in [self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]]:
                res.append(self.histogram_equalization(img, T=T))
            self.results = np.moveaxis(np.stack(res), 0, 2)

    def plot_equalize(self):
        equalized_image = self.equalize()
        plt.figure(figsize=(12, 6))
        plt.suptitle("Histogram Equalization")

        if len(self.image.shape) == 2:
            plt.subplot(2, 3, 1)
            plt.axis('off')
            plt.imshow(self.image, cmap="gray")
            plt.subplot(2, 3, 2)
            cdf = np.array(self.get_cdf(self.image))
            plt.plot(cdf)
            plt.subplot(2, 3, 3)
            plt.yticks([])
            pdf = np.diff(cdf, prepend=0)
            plt.bar(range(256), pdf)
            plt.subplot(2, 3, 4)
            plt.axis('off')
            plt.imshow(equalized_image, cmap="gray")
            plt.subplot(2, 3, 5)
            cdf = np.array(self.get_cdf(equalized_image))
            plt.plot(cdf)
            plt.subplot(2, 3, 6)
            plt.yticks([])
            pdf = np.diff(cdf, prepend=0)
            plt.bar(range(256), pdf)
            plt.show()
        else:
            plt.subplot(2, 3, 1)
            plt.axis('off')
            plt.imshow(self.image)
            plt.subplot(2, 3, 2)
            pdfs = []
            for cdf, c in zip(self.get_cdf(self.image), 'rgb'):
                plt.plot(np.array(cdf), c)
                pdfs.append((np.diff(np.array(cdf), prepend=0), c))
            plt.subplot(2, 3, 3)
            plt.yticks([])
            for pdf, c in pdfs:
                plt.bar(range(256), pdf, color=c)
            plt.subplot(2, 3, 4)
            plt.axis('off')
            plt.imshow(equalized_image)
            plt.subplot(2, 3, 5)
            pdfs = []
            for cdf, c in zip(self.get_cdf(equalized_image), 'rgb'):
                plt.plot(np.array(cdf), c)
                pdfs.append((np.diff(np.array(cdf), prepend=0), c))
            plt.subplot(2, 3, 6)
            plt.yticks([])
            for pdf, c in pdfs:
                plt.bar(range(256), pdf, color=c)
            plt.show()

    def plot_matched(self, target_image=None):
        matched_image = self.equalize() if target_image is None else self.histogram_matching(target_image)

        plt.suptitle("Histogram Matching")

        if len(self.image.shape) == 2:
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
            plt.plot(self.get_cdf(self.image), label="Source CDF")
            plt.plot(self.get_cdf(target_image), label="Target CDF", linestyle='dashed')
            plt.plot(self.get_cdf(matched_image), label="Matched CDF", linestyle='dotted')
            plt.legend()
            plt.show()
        else:
            plt.subplot(2, 2, 1)
            plt.axis('off')
            plt.imshow(self.image)
            plt.title("Source Image")
            
            plt.subplot(2, 2, 2)
            plt.axis('off')
            plt.imshow(target_image)
            plt.title("Target Image")
            
            plt.subplot(2, 2, 3)
            plt.axis('off')
            plt.imshow(matched_image)
            plt.title("Matched Image")
            
            plt.subplot(2, 2, 4)
            for i, c in enumerate(['r', 'g', 'b']):
                plt.plot(self.get_cdf(self.image)[i], label=f"Source CDF ({c})", color=c)
                plt.plot(self.get_cdf(target_image)[i], label=f"Target CDF ({c})", linestyle='dashed')
                plt.plot(self.get_cdf(matched_image)[i], label=f"Matched CDF ({c})", linestyle='dotted')
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