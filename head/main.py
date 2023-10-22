import numpy as np
import matplotlib.pyplot as plt


def grayscale(img):
    rows, cols = img.shape[0], img.shape[1]
    grayscale = np.zeros((rows, cols))

    for row in range(rows):
        for col in range(cols):
            grayscale[row, col] = np.average(img[row, col])

    return grayscale


def normalize(x, xmin=None, xmax=None):
    if xmin == None:
        xmin = np.min(x)

    if xmax == None:
        xmax = np.max(x)

    return (x - xmin) / (xmax - xmin)


def combined_filter(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    mask_diagonal = np.ones((rows, cols), np.uint8)
    r_outer = 60
    r_inner = 40
    for i in range(rows):
        for j in range(cols):
            distance_from_center = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if r_inner <= distance_from_center <= r_outer and abs(i - crow) == abs(
                j - ccol
            ):
                mask_diagonal[i, j] = 0

    mask_width = 1
    r = 60
    mask_horizontal = np.ones((rows, cols), np.uint8)
    mask_horizontal[crow - mask_width : crow + mask_width, ccol - r : ccol + r] = 0

    mask_vertical = np.ones((rows, cols), np.uint8)
    mask_vertical[crow - r : crow + r, ccol - mask_width : ccol + mask_width] = 0

    combined_mask = mask_diagonal * mask_horizontal * mask_vertical

    cradius = 2
    combined_mask[crow - cradius : crow + cradius, ccol - cradius : ccol + cradius] = 1
    masked_magnitude_spectrum = magnitude_spectrum * combined_mask

    fshift = fshift * combined_mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back, magnitude_spectrum, masked_magnitude_spectrum


img = plt.imread("HeadCT_corrupted.tif")
img = normalize(img)
img = grayscale(img)
filtered_img, mag_spectrum, masked_spectrum = combined_filter(img)

plt.figure(figsize=(18, 6))
plt.subplot(1, 4, 1), plt.imshow(img, cmap="gray"), plt.title("Original Image")
plt.subplot(1, 4, 2), plt.imshow(mag_spectrum, cmap="gray"), plt.title(
    "Initial Magnitude Spectrum"
)
plt.subplot(1, 4, 3), plt.imshow(masked_spectrum, cmap="gray"), plt.title(
    "Masked Magnitude Spectrum"
)
plt.subplot(1, 4, 4), plt.imshow(filtered_img, cmap="gray"), plt.title(
    "Image after Combined Filtering"
)
plt.tight_layout()
plt.show()
