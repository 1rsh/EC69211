# Image and Video Processing Laboratory | EC69211
**Submission By:** Irsh Vijay (21EC39055)

### How to Run:
Helper codes can be found in `fft_utils.py`. To try run `exp4.ipynb` directly.

The code contains the `FFT` and `Experiment4` class which can be used making & visualising FFTs and filtering in Frequency Domain respectively.

### Results:
#### Q1:

Ideal LPF: <br>
<p align="center">
<img src="images/lena.jpg" width="350"> <img src="outputs/lena_ideal_lpf.png" width="350"> <br>
<tr> <img src="outputs/lena_fft.png" width="350">
<img src="outputs/lena_fft_ideal_lpf.png" width="350"> </tr> <br> 
</p>
Ideal HPF: <br>
<p align="center">
<img src="images/lena.jpg" width="350"> <img src="outputs/lena_ideal_hpf.png" width="350"> 
<br>
<tr> <img src="outputs/lena_fft.png" width="350">
<img src="outputs/lena_fft_ideal_hpf.png" width="350"> </tr> <br> 
</p>
Gaussian LPF: <br>
<p align="center">
<img src="images/lena.jpg" width="350"> <img src="outputs/lena_gaussian_lpf.png" width="350"> <br>
<tr> <img src="outputs/lena_fft.png" width="350">
<img src="outputs/lena_fft_gaussian_lpf.png" width="350"> </tr> <br>
</p>
Gaussian HPF: <br>
<p align="center">
<img src="images/lena.jpg" width="350"> <img src="outputs/lena_gaussian_hpf.png" width="350"> <br>
<tr> <img src="outputs/lena_fft.png" width="350">
<img src="outputs/lena_fft_gaussian_hpf.png" width="350"> </tr> <br>
</p>
Butter LPF: <br>
<p align="center">
<img src="images/lena.jpg" width="350"> <img src="outputs/lena_butter_lpf.png" width="350"> <br>
<tr> <img src="outputs/lena_fft.png" width="350">
<img src="outputs/lena_fft_butter_lpf.png" width="350"> </tr> <br>
</p>
Butter HPF: <br>
<p align="center">
<img src="images/lena.jpg" width="350"> <img src="outputs/lena_butter_hpf.png" width="350"> <br>
<tr> <img src="outputs/lena_fft.png" width="350">
<img src="outputs/lena_fft_butter_hpf.png" width="350"> </tr> <br>
</p>

#### Q2:
The leopard_elephant image is an optical illusion which shows an elephant when viewed at low resolution and a leopard at higher resolution.
<p align="center">
<img src="outputs/leopard_elephant_down.png"> <br>
<img src="outputs/leopard_elephant_up.png">
</p>

We "hybridize" images of Albert Einstein and Marilyn Monroe.
<p align="center">
<tr> <img src="images/einstein.png"> <img src="images/marilyn.png"> </tr> <br>
</p>

Hybridizing!
<p align="center">
<img src="outputs/hybrid.png"> <br>
</p>
Then viewing at different resolutions: 
<p align="center">
<img src="outputs/einstein_marilyn_down.png"> <br>
<img src="outputs/einstein_marilyn_up.png">
</p>

#### Q3:
We have two noisy images: <br>
<p align="center">
<tr> <img src="images/cameraman_noisy1.jpg"> <img src="images/cameraman_noisy2.jpg"> </tr> <br>
</p>
Both of them have some kind of sinusoidal grating as noise, we see similar noise when we take an image of a screen. (some kind of aliasing)
<br><br>
The approach I have followed is to first pass the image through LPF and retain its content and then subtract the sinusoidal grating then add back the LPFed Image.
<p align="center">
<img src="outputs/denoise_fft_straight.png"> <br>
<img src="outputs/denoise_straight.png"> <br>
</p>
Similar approach can be followed for `cameraman_noisy2.jpg`
<p align="center">
<img src="outputs/denoise_fft_diag.png"> <br>
<img src="outputs/denoise_diag.png"> 
</p>
We can also have something like:
<p align="center">
<img src="outputs/denoise_fft_angle.png"> <br>
<img src="outputs/denoise_angle.png">  <br>
<img src="outputs/denoise_fft_multi.png"> <br>
<img src="outputs/denoise_multi.png"> 
</p>