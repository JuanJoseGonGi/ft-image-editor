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

    mask_width = 1

    mask_diagonal = np.ones((rows, cols), np.uint8)

    offsets = []
    for offset in offsets:
        for row in range(rows):
            for col in range(cols):
                if abs(row - crow) == abs(col - ccol) + offset:
                    rb = row - mask_width
                    rt = row + mask_width
                    cb = col - mask_width
                    ct = col + mask_width
                    mask_diagonal[rb:rt, cb:ct] = 0

    mask_horizontal = np.ones((rows, cols), np.uint8)

    offsets = [0, 39, 54, 66, 85, 92]
    for offset in offsets:
        mask_horizontal[crow - offset - mask_width : crow - offset + mask_width, :] = 0
        mask_horizontal[crow + offset - mask_width : crow + offset + mask_width, :] = 0

    mask_vertical = np.ones((rows, cols), np.uint8)

    offsets = [0, 54, 70, 75, 80, 100]
    for offset in offsets:
        mask_vertical[:, ccol - offset - mask_width : ccol - offset + mask_width] = 0
        mask_vertical[:, ccol + offset - mask_width : ccol + offset + mask_width] = 0

    combined_mask = mask_diagonal * mask_horizontal * mask_vertical

    cradius = 4
    combined_mask[crow - cradius : crow + cradius, ccol - cradius : ccol + cradius] = 1
    masked_magnitude_spectrum = magnitude_spectrum * combined_mask

    fshift = fshift * combined_mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back, magnitude_spectrum, masked_magnitude_spectrum


img = plt.imread("ruido9.jpg")
img = normalize(img)
img = grayscale(img)

plt.figure(figsize=(18, 6))
plt.subplot(1, 4, 1), plt.imshow(img, cmap="gray"), plt.title("Original Image")
filtered_img, mag_spectrum, masked_spectrum = combined_filter(img)
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
