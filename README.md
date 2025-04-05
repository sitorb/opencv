# Image Processing Techniques with OpenCV and Python

This project demonstrates various image processing techniques using OpenCV and Python. It covers essential operations such as grayscale conversion, blurring, sharpening, edge detection, and color channel histograms.

## Project Structure

The project consists of a single Python script (`image_processing.py`) that implements all the image processing operations. The script is organized into sections for each technique, with comments explaining the code.

## Code Explanation

### 1. Importing Libraries

The script begins by importing necessary libraries:
python import cv2 from matplotlib import pyplot as plt import numpy as np
- `cv2`: OpenCV library for image processing.
- `matplotlib.pyplot`: Library for plotting images and graphs.
- `numpy`: Library for numerical operations, especially for working with arrays.

### 2. Loading and Displaying the Image

The image is loaded using `cv2.imread()` and displayed using `plt.imshow()`:
python image = cv2.imread('gb.jpg') image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) plt.imshow(image) plt.axis('off') plt.title('Original Image') plt.show()

### 3. Grayscale Conversion

The image is converted to grayscale using `cv2.cvtColor()`:
python gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

### 4. Blurring

Two types of blurring are applied: Mean Blur and Gaussian Blur.

- **Mean Blur:** Averages pixel values within a kernel.
- **Gaussian Blur:** Weights pixel values based on a Gaussian distribution.
python mean_blur = cv2.blur(gray_image, (3, 3)) gaussian_blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
### 5. Sharpening

A sharpening kernel is applied using `cv2.filter2D()`:
python kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) sharpened_image = cv2.filter2D(gray_image, -1, kernel)

### 6. Edge Detection

Two edge detection methods are used: Sobel and Canny.

- **Sobel:** Calculates gradients in x and y directions to detect edges.
- **Canny:** Uses multi-stage algorithm to detect edges with low error rate.
python sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3) sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3) sobel_edges = cv2.magnitude(sobel_x, sobel_y)

canny_edges = cv2.Canny(gray_image, 100, 200)
### 7. Color Channel Histograms

Histograms are computed for each color channel (R, G, B) using `cv2.calcHist()` and plotted using `plt.plot()`:
python r_channel, g_channel, b_channel = cv2.split(image) r_hist = cv2.calcHist([r_channel], [0], None, [256], [0, 256]) g_hist = cv2.calcHist([g_channel], [0], None, [256], [0, 256]) b_hist = cv2.calcHist([b_channel], [0], None, [256], [0, 256])


## Logic and Algorithms

The project applies fundamental image processing algorithms:

- **Grayscale Conversion:** Reduces color information to intensity.
- **Blurring:** Smooths the image by averaging pixel values.
- **Sharpening:** Enhances edges by increasing contrast.
- **Edge Detection:** Identifies boundaries between objects in the image.
- **Color Channel Histograms:** Represents the distribution of pixel intensities for each color channel.
  
![image](https://github.com/user-attachments/assets/fb6dc8c7-6ab1-46a4-b70a-9a7215ffde53)

## Technology

- **OpenCV:** A powerful library for computer vision tasks.
- **Python:** A versatile programming language for scientific computing.
- **Matplotlib:** A library for creating visualizations.
- **NumPy:** A library for numerical operations.

![image](https://github.com/user-attachments/assets/a8aacb54-dbe7-468f-bd6e-1526fbbc1602)


## Conclusion

This project showcases basic image processing techniques using OpenCV and Python. It provides a foundation for more advanced image analysis and manipulation tasks.
