
# Image and Video Processing Laboratory | EC69211
**Submission By:** Irsh Vijay (21EC39055)

### How to Run:
Helper codes can be found in `utils/`. To try run `exp2.ipynb` directly.

The code contains the `BMP` class which can be used reading, writing and modifying channels of bitmap images. 

**How to Run:**
Reading
```python
bmp = BMP()
bmp.read(filepath) # reads the file
bmp.summary() # prints the summary
bmp.plot() # plots the image
```

Writing
```python
bmp = BMP()
bmp.write(filepath, pixel_array, bits_per_pixel) # writes the image
```

Colour Channel Modification
```python
bmp = BMP()
bmp.read(filepath) # reads the file
bmp.permute_channels(channel = 'BRG', zero = 'R') # some anagram of 'RGB'
bmp.plot()
```
