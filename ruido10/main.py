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

    # mask_lowpass = np.zeros((rows, cols), np.uint8)
    # r = 70
    # for row in range(rows):
    #     for col in range(cols):
    #         dist = np.sqrt((row - crow) ** 2 + (col - ccol) ** 2)
    #         if dist <= r:
    #             mask_lowpass[row, col] = 1

    mask_width = 1
    mask_diagonal = np.ones((rows, cols), np.uint8)

    angle_error = 1
    angles = [24, 45, 135, 165]
    for row in range(rows):
        for col in range(cols):
            rads = -np.arctan2(row - crow, col - ccol)
            degrees = np.rad2deg(rads)
            # normalize to first quadrant to apply to all quadrants
            if degrees < 0:
                degrees = 180 + degrees

            for angle in angles:
                if degrees > angle - angle_error and degrees < angle + angle_error:
                    rb = row - mask_width
                    rt = row + mask_width
                    cl = col - mask_width
                    cr = col + mask_width

                    mask_diagonal[rb:rt, cl:cr] = 0

    mask_horizontal = np.ones((rows, cols), np.uint8)

    offsets = [3]
    for offset in offsets:
        mask_horizontal[crow - offset - mask_width : crow - offset + mask_width, :] = 0
        mask_horizontal[crow + offset - mask_width : crow + offset + mask_width, :] = 0

    mask_vertical = np.ones((rows, cols), np.uint8)

    offsets = [0, 7]
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


img = plt.imread("ruido10.jpg")
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
