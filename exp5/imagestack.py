from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import multiprocess as mp
import concurrent.futures

class ImageStack:
    def __init__(self, stackpath) -> None:
        self.stackpath = stackpath
        self.stack, self.names = self.read_images(stackpath)
        self.available_filters = ["mean", "median", "prewitt_x", "prewitt_y", "sobel_x", "sobel_y", "laplacian", "gaussian", "log"]
        
        self.num_cores = mp.cpu_count() - 1
        self.pool = mp.Pool(self.num_cores)
        
        self.cache = {k: [] for k in self.available_filters}
        # self.prepare_cache()
        self.cache_preparation_future = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_cores).submit(self.prepare_cache)

    def apply_filter_wrapper(self, args):
        if len(args) == 3:
            img_idx, filter_name, filter_args = args
            return self.apply_filter(self.stack[img_idx], filter_name, filter_args)
        else:
            img_idx, filter_name = args
            return self.apply_filter(self.stack[img_idx], filter_name)
        
    def prepare_cache(self):
        self.valid_args = {"mean": [{"filter_size": i} for i in range(1, 8, 2)], "median": [{"filter_size": i} for i in range(1, 8, 2)], "gaussian": [{"filter_size": i, "sigma": j} for i in range(1, 8, 2) for j in range(1, 8, 2)]}

        tasks = []
        for filter_name in self.available_filters:
            for img_idx, img in enumerate(self.stack):
                if filter_name in self.valid_args:
                    for args in self.valid_args[filter_name]:
                        if type(self.cache[filter_name]) == list:
                            self.cache[filter_name] = {}
                        self.cache[filter_name][str(args)] = []
                        tasks.append((img_idx, filter_name, args))
                else:
                    tasks.append((img_idx, filter_name))

        results = self.pool.map(self.apply_filter_wrapper, tasks)

        for i, args in enumerate(tasks):
            filter_name = args[1]
            
            if filter_name in ["mean", "median", "gaussian"]:
                self.cache[filter_name][str(args[2])].append(results[i])
            else:
                self.cache[filter_name].append(results[i])


        self.pool.close()
        self.pool.join()

    def read_image(self, filepath):
        return np.asarray(Image.open(filepath))

    def read_images(self, stackpath):
        stack = []
        names = []
        for filepath in os.listdir(stackpath):
            if filepath[-4:] in {".jpg", ".png", ".bmp"}:
                path = os.path.join(stackpath, filepath)
                names.append(filepath[:-4])
                stack.append(self.read_image(path))
        
        return stack, names
            
    def convolve(self, image, kernel, clip=True):
        image = np.array(image)
        
        if type(kernel) == int:
            k_size = kernel
            output = np.zeros((image.shape[0] - k_size + 1, image.shape[1]  - k_size + 1))

            for i in range(image.shape[0] - k_size + 1):
                for j in range(image.shape[1] - k_size + 1):
                    region = image[i:i+k_size, j:j+k_size]
                    conv_sum = np.median(region)
                    output[i, j] = conv_sum
                    if clip:
                        output[i, j] = np.clip(output[i, j], 0, 255)
                
            return output
        
        kernel = np.array(kernel)
        k_size = kernel.shape[0]
        image = np.pad(image, pad_width=((k_size//2, k_size//2), (k_size//2, k_size//2)), mode='constant', constant_values=0)

        output = np.zeros((image.shape[0] - k_size + 1, image.shape[1]  - k_size + 1))
        
        for i in range(image.shape[0] - k_size + 1):
            for j in range(image.shape[1] - k_size + 1):
                region = image[i:i+k_size, j:j+k_size]
                conv_sum = np.sum(region * kernel)
                output[i, j] = conv_sum
                if clip:
                    output[i, j] = np.clip(output[i, j], 0, 255)
        
        return output
    
    def construct_filter(self, name, filter_args=None):
        if name == "mean":
            filter_size = filter_args["filter_size"]
            return np.ones((filter_size, filter_size)) / (filter_size ** 2)
        elif name == "median":
            return filter_args["filter_size"]
        elif name == "prewitt_x":
            return np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]])
        elif name == "prewitt_y":
            return np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1, -1, -1]])
        elif name == "sobel_x":
            return np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        elif name == "sobel_y":
            return np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        elif name == "laplacian":
            return np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])
        elif name == "gaussian":
            sigma = filter_args["sigma"]
            filter_size = filter_args["filter_size"]

            ax = np.linspace(-(filter_size // 2), filter_size // 2, filter_size)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

            return kernel / np.sum(kernel)
        elif name == "log":
            return np.array([[0, 0, 1, 0, 0],
                            [0, 1, 2, 1, 0],
                            [1, 2, -16, 2, 1],
                            [0, 1, 2, 1, 0],
                            [0, 0, 1, 0, 0]])

        else:
            raise ValueError(f"Filter '{name}' not recognized or unsupported for size {filter_size}")
        
    
    def gaussian_unblur(self, image, filter_args, epsilon = 1e-3):
        Ik = np.copy(image)

        G_sigma = self.construct_filter("gaussian", filter_args)
        prev_Ik = np.zeros_like(image)

        k = 0

        criterion = abs(Ik - prev_Ik).mean() / 255

        while criterion > epsilon:
            print(f"Iteration: {k+1}, L1 Loss: {criterion:.4f}", end="\r")
            Ak = self.convolve(Ik, G_sigma)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                Bk = np.where(Ak != 0, image / Ak, 0) 
            
            Ck = self.convolve(Bk, G_sigma)
            
            prev_Ik = Ik
            Ik = Ik * Ck
            criterion = abs(Ik - prev_Ik).mean() / 255
            k+=1

        return Ik, k
        
    def apply_filter(self, image, name, filter_args=None, image_idx=-1):
        if image_idx == -1:
            kernel = self.construct_filter(name, filter_args)
            return self.convolve(image, kernel)
        else:
            try:
                if name in ["mean", "median", "gaussian"]:
                    return self.cache[name][str(filter_args)][image_idx]
                else:
                    return self.cache[name][image_idx]
            except:
                print("Cache is being prepared in parallel! Reverting to computing in callback!")
                kernel = self.construct_filter(name, filter_args)
                return self.convolve(self.stack[image_idx], kernel)

    def plot_transformed(self, image, name, filter_args=None, save=False, image_idx=-1):

        if not(save):
            if image_idx!=-1:
                image = self.stack[image_idx]
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap="gray")
            plt.title("Original")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(Image.fromarray(self.apply_filter(image, name, filter_args, image_idx=image_idx)))
            plt.title(f"{name.title()} Filter")
            plt.axis('off')
        else:
            Image.fromarray(self.apply_filter(image, name, filter_args)).convert('L').save(save)

    def plot_gaussian_unblur(self, image, filter_args, epsilon):
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap="gray")
        plt.title("Original")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        blurred_image = self.apply_filter(image, "gaussian", filter_args)
        plt.imshow(blurred_image, cmap="gray")
        plt.title(f"Blurred Image")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        recovered_image, k = self.gaussian_unblur(blurred_image, filter_args, epsilon)
        plt.imshow(recovered_image, cmap="gray")
        plt.title(f"Recovered Image: {k} iters")
        plt.axis('off')


    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save_all(self, savepath):
        for i in tqdm(range(len(self.stack))):
            for f in ["mean", "median", "prewitt_x", "prewitt_y", "sobel_x", "sobel_y", "laplacian", "gaussian", "log"]:
                if f in ["mean", "median", "gaussian"]:
                    for filter_size in [1, 3, 5, 7]:
                        filter_args = {"filter_size": filter_size}
                        if f == "gaussian":
                            for sigma in range(1, filter_size+1, 2):
                                filter_args["sigma"] = sigma
                                self.plot_transformed(self.stack[i], f, filter_args, save=f"{savepath}{self.names[i]}_{f}_{str(filter_args)}.png")
                        else:
                            self.plot_transformed(self.stack[i], f, filter_args, save=f"{savepath}{self.names[i]}_{f}_{str(filter_args)}.png")
                else:
                    self.plot_transformed(self.stack[i], f, filter_args, save=f"{savepath}{self.names[i]}_{f}_{str(filter_args)}.png")



class ImageStackLazy:
    def __init__(self, stackpath) -> None:
        self.stackpath = stackpath
        self.stack, self.names = self.read_images(stackpath)
        self.available_filters = ["mean", "median", "prewitt_x", "prewitt_y", "sobel_x", "sobel_y", "laplacian", "gaussian", "log"]
        
    def read_image(self, filepath):
        return np.asarray(Image.open(filepath))

    def read_images(self, stackpath):
        stack = []
        names = []
        for filepath in os.listdir(stackpath):
            if filepath[-4:] in {".jpg", ".png", ".bmp"}:
                path = os.path.join(stackpath, filepath)
                names.append(filepath[:-4])
                stack.append(self.read_image(path))
        
        return stack, names
            
    def convolve(self, image, kernel, clip=True):
        image = np.array(image)
        
        if type(kernel) == int:
            k_size = kernel
            output = np.zeros((image.shape[0] - k_size + 1, image.shape[1]  - k_size + 1))

            for i in range(image.shape[0] - k_size + 1):
                for j in range(image.shape[1] - k_size + 1):
                    region = image[i:i+k_size, j:j+k_size]
                    conv_sum = np.median(region)
                    output[i, j] = conv_sum
                    if clip:
                        output[i, j] = np.clip(output[i, j], 0, 255)
                
            return output
        
        kernel = np.array(kernel)
        k_size = kernel.shape[0]
        image = np.pad(image, pad_width=((k_size//2, k_size//2), (k_size//2, k_size//2)), mode='constant', constant_values=0)

        output = np.zeros((image.shape[0] - k_size + 1, image.shape[1]  - k_size + 1))
        
        for i in range(image.shape[0] - k_size + 1):
            for j in range(image.shape[1] - k_size + 1):
                region = image[i:i+k_size, j:j+k_size]
                conv_sum = np.sum(region * kernel)
                output[i, j] = conv_sum
                if clip:
                    output[i, j] = np.clip(output[i, j], 0, 255)
        
        return output
    
    def construct_filter(self, name, filter_args=None):
        if name == "mean":
            filter_size = filter_args["filter_size"]
            return np.ones((filter_size, filter_size)) / (filter_size ** 2)
        elif name == "median":
            return filter_args["filter_size"]
        elif name == "prewitt_x":
            return np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]])
        elif name == "prewitt_y":
            return np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1, -1, -1]])
        elif name == "sobel_x":
            return np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        elif name == "sobel_y":
            return np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        elif name == "laplacian":
            return np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])
        elif name == "gaussian":
            sigma = filter_args["sigma"]
            filter_size = filter_args["filter_size"]

            ax = np.linspace(-(filter_size // 2), filter_size // 2, filter_size)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

            return kernel / np.sum(kernel)
        elif name == "log":
            return np.array([[0, 0, 1, 0, 0],
                            [0, 1, 2, 1, 0],
                            [1, 2, -16, 2, 1],
                            [0, 1, 2, 1, 0],
                            [0, 0, 1, 0, 0]])

        else:
            raise ValueError(f"Filter '{name}' not recognized or unsupported for size {filter_size}")
        
    
    def gaussian_unblur(self, image, filter_args, epsilon = 1e-3):
        Ik = np.copy(image)

        G_sigma = self.construct_filter("gaussian", filter_args)
        prev_Ik = np.zeros_like(image)

        k = 0

        while (Ik - prev_Ik).max() > epsilon * 255:
            Ak = self.convolve(Ik, G_sigma)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                Bk = np.where(Ak != 0, image / Ak, 0) 
            
            Ck = self.convolve(Bk, G_sigma)
            
            prev_Ik = Ik
            Ik = Ik * Ck
            k+=1

        return Ik, k
        
    def apply_filter(self, image, name, filter_args=None, image_idx=-1):
        kernel = self.construct_filter(name, filter_args)
        return self.convolve(image, kernel)

    def plot_transformed(self, image, name, filter_args=None, save=False, image_idx=-1):

        if not(save):
            if image_idx!=-1:
                image = self.stack[image_idx]
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap="gray")
            plt.title("Original")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(Image.fromarray(self.apply_filter(image, name, filter_args, image_idx=image_idx)))
            plt.title(f"{name.title()} Filter")
            plt.axis('off')
        else:
            Image.fromarray(self.apply_filter(image, name, filter_args)).convert('L').save(save)

    def plot_gaussian_unblur(self, image, filter_args, epsilon):
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap="gray")
        plt.title("Original")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        blurred_image = self.apply_filter(image, "gaussian", filter_args)
        plt.imshow(blurred_image, cmap="gray")
        plt.title(f"Blurred Image")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        recovered_image, k = self.gaussian_unblur(blurred_image, filter_args, epsilon)
        plt.imshow(recovered_image, cmap="gray")
        plt.title(f"Recovered Image: {k} iters")
        plt.axis('off')

    def save_all(self, savepath):
        for i in tqdm(range(len(self.stack))):
            for f in ["mean", "median", "prewitt_x", "prewitt_y", "sobel_x", "sobel_y", "laplacian", "gaussian", "log"]:
                if f in ["mean", "median", "gaussian"]:
                    for filter_size in [1, 3, 5, 7]:
                        filter_args = {"filter_size": filter_size}
                        if f == "gaussian":
                            for sigma in range(1, filter_size+1, 2):
                                filter_args["sigma"] = sigma
                                self.plot_transformed(self.stack[i], f, filter_args, save=f"{savepath}{self.names[i]}_{f}_{str(filter_args)}.png")
                        else:
                            self.plot_transformed(self.stack[i], f, filter_args, save=f"{savepath}{self.names[i]}_{f}_{str(filter_args)}.png")
                else:
                    self.plot_transformed(self.stack[i], f, filter_args, save=f"{savepath}{self.names[i]}_{f}_{str(filter_args)}.png")




