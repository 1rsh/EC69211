# Image and Video Processing Laboratory | EC69211
**Submission By:** Irsh Vijay (21EC39055)

### How to Run:
To try run `exp5.ipynb` directly.

The code contains the `ImageStack` class in the `imagestack.py` and the `UI` class in `ui.py` to be used for spatial domain filtering (convolutions). 

Any `ImageStack` object takes in a folder path and starts computing cache asynchronously using n-1 CPU cores (all possible convolution requests). It takes about 35-40s for the cache to prepare for 11 images on 14 CPU cores. Also, the class is usable in the meantime (convolutions are computed in the callback during this time). This has been done so that user can still see outputs without waiting for the whole cache to form.

### Effects of different filters:
* Mean:
<br>
<img src="saved_normal_images/jetplane_mean_{&apos;filter_size&apos;: 7}.png"></img>
* Median:
<br>
<img src="saved_normal_images/jetplane_median_{&apos;filter_size&apos;: 7}.png"></img>
* Prewitt X:
<br>
<img src="saved_normal_images/jetplane_prewitt_x_{&apos;filter_size&apos;: 7}.png"></img>
* Prewitt Y:
<br>
<img src="saved_normal_images/jetplane_prewitt_y_{&apos;filter_size&apos;: 7}.png"></img>
* Sobel X:
<br>
<img src="saved_normal_images/jetplane_sobel_x_{&apos;filter_size&apos;: 7}.png"></img>
* Sobel Y:
<br>
<img src="saved_normal_images/jetplane_sobel_y_{&apos;filter_size&apos;: 7}.png"></img>
* Laplacian:
<br>
<img src="saved_normal_images/jetplane_laplacian_{&apos;filter_size&apos;: 7}.png"></img>
* Gaussian:
<br>
<img src="saved_normal_images/jetplane_gaussian_{&apos;filter_size&apos;: 7, &apos;sigma&apos;: 7}.png"></img>
* Laplacian of Gaussian:
<br>
<img src="saved_normal_images/jetplane_log_{&apos;filter_size&apos;: 7}.png"></img>

### Gaussian Unblur:
<br>
<img src="assets/gaussian_unblur.png"></img>