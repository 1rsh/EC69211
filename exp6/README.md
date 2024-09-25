# Image and Video Processing Laboratory | EC69211 | Experiment - 6
**Submission By:** Irsh Vijay (21EC39055)

### How to Run:
Individual codes can be found in `histogram_utils.py`. To try run `exp6.py`:

Setup:
```shell
pip install -r requirements.txt
```

Histogram Equalization:
```shell
python exp6.py -s "path/to/image.png"
```

Histogram Matching:
```shell
python exp6.py -s "path/to/source/image.png" -t "path/to/target/image.png"
```

Precomputed outputs are saved to `output/`.

### Code Structure:
- `exp6.py`: Main script to plot histogram equalization or matching.
- `histogram_utils.py`: Contains the Histogram class with all key functions:
- `histogram_equalization()`: Performs histogram equalization.
- `histogram_matching()`: Performs histogram matching with a custom transformation.
- `calculate_histogram()`, `calculate_cdf()`: Utility functions to calculate the histogram and CDF.