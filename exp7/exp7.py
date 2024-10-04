import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython import display
import argparse
import os

class Exp7Image:
    def __init__(self, filepath=None, data=None):
        if filepath:
            self.filepath = filepath
            self.load_image()
        else:
            self.image = data
        
        self.image = self.image.astype(np.bool_)

    def load_image(self):
        self.image = np.asarray(Image.open(self.filepath))

    def morphological_operation(self, image, struct_elem, operation):
        img_rows, img_cols = image.shape
        elem_rows, elem_cols = struct_elem.shape
        
        padded_image = np.pad(image, (elem_rows // 2, elem_cols // 2), mode='constant')
        
        output = np.zeros_like(image)
        
        for i in range(img_rows):
            for j in range(img_cols):
                region = padded_image[i:i + elem_rows, j:j + elem_cols]

                if operation(region, struct_elem):
                    output[i, j] = 1

        return Exp7Image(data=output)
    
    def erode(self, B):
        B = B.astype(np.bool_)
        return self.morphological_operation(self.image, B, lambda a, b: np.all(np.where(b, a & b, 1)))
    
    def dilate(self, B):
        B = B.astype(np.bool_)
        return self.morphological_operation(self.image, B, lambda a, b: np.any(a & b))
    
    def open(self, B):
        B = B.astype(np.bool_)
        return self.erode(B).dilate(B)
    
    def close(self, B):
        B = B.astype(np.bool_)
        return self.dilate(B).erode(B)
    
    def _repr_png_(self):
        return display.display(Image.fromarray(self.image))
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="Image Filepath")
    print("\033[92m"+"="*os.get_terminal_size().columns)
    print(f"\nImage and Video Processing Laboratory | EC69211 | Experiment - 7")
    print("\nCode Author: Irsh Vijay | 21EC39055\n")
    print("-"*os.get_terminal_size().columns + "\n\033[0m")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = get_args()

    Bs = []
    Bs.append(np.ones((1, 2)))
    Bs.append(np.ones((3, 3)))
    Bs.append(np.ones((9, 9)))
    Bs.append(np.ones((15, 15)))
    Bs.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    idx = int(input("Choose Structuring Element:\n\n0: (1, 2)\n1: (3, 3)\n2: (9, 9)\n3: (15, 15)\n4: plus\n\nChoice: "))

    B = Bs[idx]

    A = Exp7Image(filepath=args.f)

    # A = Exp7Image(data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 1, 1, 1, 0, 0],
    #                        [0, 0, 1, 0, 1, 1, 0, 0],
    #                        [0, 1, 1, 0, 1, 0, 0, 0],
    #                        [0, 1, 1, 1, 0, 0, 0, 0],
    #                        [0, 1, 1, 1, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 0, 0, 0, 0]]).repeat(2, 0).repeat(2, 1))
    # A = Exp7Image(data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 1, 1, 1, 0, 0],
    #                        [0, 0, 0, 1, 1, 1, 0, 0],
    #                        [0, 0, 0, 1, 1, 1, 0, 0],
    #                        [0, 0, 0, 0, 1, 0, 0, 0],
    #                        [0, 0, 0, 1, 1, 1, 0, 0],
    #                        [0, 0, 0, 1, 1, 1, 0, 0],
    #                        [0, 0, 0, 0, 0, 0, 0, 0]]).repeat(2, 0).repeat(2, 1))

    # plotting
    cmap = ListedColormap(["black", "white"])
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(A.image, cmap=cmap)
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.title("Structuring Element")
    plt.imshow(np.pad(B, 1), cmap=cmap)
    plt.subplot(2, 3, 3)
    plt.title("Eroded Image")
    plt.imshow(A.erode(B).image, cmap=cmap)
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.title("Dilated Image")
    plt.imshow(A.dilate(B).image, cmap=cmap)
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.title("Opened Image")
    plt.imshow(A.open(B).image, cmap=cmap)
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.title("Closed Image")
    plt.imshow(A.close(B).image, cmap=cmap)
    plt.axis('off')
    
    plt.show()

    print("\033[92m"+"-"*os.get_terminal_size().columns + "\n\033[0m")