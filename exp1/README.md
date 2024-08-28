
# Image and Video Processing Laboratory | EC69211
**Submission By:** Irsh Vijay (21EC39055)

### How to Run:
Individual codes can be found in `utils.py`. To try run `exp1.py`:

Please run
```shell
pip install -r requirements.txt
```

Scaling:
```shell
python exp1.py f="path/to/image.png" -s=<scale_factor> -t="bilinear"
```

Rotation:
```shell
python exp1.py f="path/to/image.png" -r=<rotation_angle> -t="nearest"
```

Output will be saved to `output/`.


### Directory Structure:
```
¦   exp1.py
¦   hawa-mahal.png
¦   README.md
¦   requirements.txt
¦   utils.py
¦   
+---output
¦       rotate60_bilinear.png
¦       rotate60_nearest.png
¦       scale2_bilinear.png
¦       scale2_nearest.png
```

The code provides two functions for manipulating images: `scale_image` and `rotate_image`. Both functions support two types of interpolation: "nearest" and "bilinear."

### Interpolation Techniques (with `scale_image` and `rotate_image`):

-   **Nearest Neighbor Interpolation**:
    
    -   For each pixel in the scaled image, it finds the corresponding pixel in the original image by dividing the coordinates by the scaling factor (`k`).
    -   The value of the pixel is directly taken from the nearest neighbor in the original image.
-   **Bilinear Interpolation**:
    
    -   For each pixel in the scaled image, it calculates the position in the original image as a float.
    -   It uses the four nearest neighboring pixels in the original image to compute a weighted average based on the distances from these neighboring pixels.
    -   The final pixel value is a blend of these neighboring pixels, providing smoother scaling compared to nearest neighbor interpolation.