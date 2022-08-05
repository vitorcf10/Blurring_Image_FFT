import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import cmath
from scipy.signal import convolve, correlate

#
# Step 4 - generating low-pass filter in frequency domain
# 
def ideal_low_pass_filter(img, radius):
    '''Cria um filtro passa-baixa de mesma dimensão que img. 
       radius define o raio do filtro, isto é, a máxima frequência que será mantida na imagem'''
    
    num_rows, num_cols = img.shape
    freq_r = np.fft.fftfreq(num_rows)
    freq_c = np.fft.fftfreq(num_cols)
    freq_r = np.fft.fftshift(freq_r)
    freq_c = np.fft.fftshift(freq_c)  
    
    low_pass_filter = np.zeros([num_rows, num_cols])
    for row in range(num_rows):
        for col in range(num_cols):
            distance = np.sqrt(freq_r[row]**2 + freq_c[col]**2)
            if distance <= radius:
                low_pass_filter[row, col] = 1
                
    return low_pass_filter

# read and print original image(building.jpg)
img = cv2.imread('predio.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=[7,7])
plt.imshow(img, cmap='gray')
plt.show()

#
# Step 1 - image padding 
#

img_rows, img_cols = img.shape

img_padded = np.zeros((512, 512), dtype=img.dtype)
for i in range(img_rows):
  for j in range(img_cols):
    img_padded[i][j] = img[i][j]

img_fft_nshift = np.fft.fft2(img_padded)

#
# Step 2 - Multiply padded image * (-1)^(x+y): in order to shift the fourier spectrum.
# 

for i in range(img_rows):
  for j in range(img_cols):
    img_padded[i][j] *= (-1)**(i+j)

#
# Step 3 - Convert image from space domain to frequency domain using fft
#

img_fft_shifted = np.fft.fft2(img_padded)
rimg = np.fft.ifft2(img_fft_shifted)

magnitude_antes = np.zeros((img_rows, img_cols))

plt.figure(figsize = (12, 12))
# Magnitude before
magnitude_antes = np.abs(img_fft_shifted)
plt.subplot(141), plt.imshow(magnitude_antes, cmap = 'gray')
# Magnitude after
magnitude_depois =  30*np.log(np.abs(img_fft_shifted)+1)
plt.subplot(142), plt.imshow(magnitude_depois, cmap = 'gray')

# Phase
fase = np.arctan(img_fft_shifted.imag/img_fft_shifted.real)
plt.subplot(143), plt.imshow(fase, cmap = 'gray')

plt.show()

plt.figure(figsize = (12, 12))

magnitude_antes_nshift = np.abs(img_fft_nshift)
magnitude_sem_shift =  30*np.log(np.abs(img_fft_nshift)+1)
# Shifted magnitude
plt.subplot(145), plt.imshow(magnitude_depois, cmap = 'gray')
# Non shifted magnitude
plt.subplot(146), plt.imshow(magnitude_sem_shift, cmap = 'gray')

# Blurring image multiplying the magnitude spectrum and generated filter all in frequency domain
H = ideal_low_pass_filter(img_padded,0.07)
G = img_fft_shifted*H
# Going back to space domain
filtered_img = np.fft.ifft2(G)
plt.figure(figsize = (12, 12))
# Take off shift
for i in range(img_rows):
  for j in range(img_cols):
    filtered_img[i][j] *= (-1)**(i+j)
real_img = np.real(filtered_img)
# Print blur image
plt.imshow(real_img, cmap = 'gray')




