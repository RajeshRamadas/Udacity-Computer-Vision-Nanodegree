# Introduction to Computer Vision
## Image Representation and Processing Fundamentals

### Table of Contents
1. [Digital Image Representation](#digital-image-representation)
2. [Grayscale Images](#grayscale-images)
3. [Color Images (RGB)](#color-images-rgb)
4. [Image Coordinate Systems](#image-coordinate-systems)
5. [Basic Image Properties](#basic-image-properties)
6. [Fundamental Image Operations](#fundamental-image-operations)
7. [Code Examples](#code-examples)
8. [Practical Exercises](#practical-exercises)
9. [Key Takeaways](#key-takeaways)

---

## Digital Image Representation

Images are represented as **numerical arrays** in computers:
- Each pixel contains intensity values
- Spatial arrangement preserves visual information
- All computer vision algorithms operate on these numerical representations

### Why This Matters
- Computers can't "see" like humans - they work with numbers
- Every CV algorithm manipulates pixel values mathematically
- Understanding representation is crucial for preprocessing and analysis

---

## Grayscale Images

### Structure
- **2D array/matrix** of pixel intensities
- **Single channel** (one value per pixel)
- **Value range**: 0-255 (8-bit) or 0.0-1.0 (normalized)

### Intensity Mapping
| Value | Appearance | Description |
|-------|------------|-------------|
| 0     | Black      | No light    |
| 128   | Gray       | Medium intensity |
| 255   | White      | Maximum intensity |

### Mathematical Representation
```
Grayscale Image = Matrix[Height × Width]

Example 3×3 image:
[  0  128  255]    [Black  Gray   White]
[ 64  192   32] => [Dark   Light  Dark ]  
[255    0  128]    [White  Black  Gray ]
```

### Memory Requirements
- **Formula**: Height × Width × 1 byte
- **Example**: 1920×1080 image = 2,073,600 bytes ≈ 2MB

---

## Color Images (RGB)

### Structure
- **3D array**: Height × Width × 3 channels
- **Three channels**: Red, Green, Blue
- **Each channel**: 0-255 intensity values

### Channel Representation
```
RGB Image = Matrix[Height × Width × 3]

Single pixel: [R, G, B]
- [255,   0,   0] = Pure Red
- [  0, 255,   0] = Pure Green  
- [  0,   0, 255] = Pure Blue
- [255, 255, 255] = White
- [  0,   0,   0] = Black
- [255, 255,   0] = Yellow (Red + Green)
```

### Color Mixing Examples
| RGB Values | Color | Description |
|------------|-------|-------------|
| [255, 0, 0] | Red | Maximum red, no green/blue |
| [0, 255, 0] | Green | Maximum green, no red/blue |
| [0, 0, 255] | Blue | Maximum blue, no red/green |
| [255, 255, 0] | Yellow | Red + Green mixing |
| [255, 0, 255] | Magenta | Red + Blue mixing |
| [0, 255, 255] | Cyan | Green + Blue mixing |
| [128, 128, 128] | Gray | Equal RGB values |

### Memory Requirements
- **Formula**: Height × Width × 3 bytes
- **Example**: 1920×1080 RGB image = 6,220,800 bytes ≈ 6MB

---

## Image Coordinate Systems

### Standard Convention
```
(0,0) ---------> X (Width/Columns)
  |
  |
  |
  v
Y (Height/Rows)
```

### Key Points
- **Origin**: Top-left corner (0, 0)
- **X-axis**: Horizontal (left to right)
- **Y-axis**: Vertical (top to bottom)
- **Pixel access**: `image[y, x]` or `image[row, column]`

### Coordinate Examples
```
For a 5×5 image:
(0,0) (0,1) (0,2) (0,3) (0,4)
(1,0) (1,1) (1,2) (1,3) (1,4)
(2,0) (2,1) (2,2) (2,3) (2,4)  ← Center: (2,2)
(3,0) (3,1) (3,2) (3,3) (3,4)
(4,0) (4,1) (4,2) (4,3) (4,4)  ← Bottom-right: (4,4)
```

---

## Basic Image Properties

### Resolution
- **Definition**: Number of pixels in image
- **Format**: Width × Height (e.g., 1920×1080)
- **Total pixels**: Width × Height

### Bit Depth
- **8-bit**: 256 possible values (0-255)
- **16-bit**: 65,536 possible values
- **32-bit**: 4+ billion possible values

### Aspect Ratio
- **Definition**: Width ÷ Height
- **Common ratios**: 16:9, 4:3, 1:1
- **Example**: 1920×1080 = 16:9 ratio

### File Size Calculation
```
Uncompressed Size = Width × Height × Channels × (Bits per pixel / 8)

Examples:
- Grayscale 100×100: 100 × 100 × 1 × 1 = 10,000 bytes
- RGB 100×100: 100 × 100 × 3 × 1 = 30,000 bytes
- RGB 1920×1080: 1920 × 1080 × 3 × 1 = 6,220,800 bytes
```

---

## Fundamental Image Operations

### 1. Point Operations (Pixel-wise)

#### Brightness Adjustment
```python
# Increase brightness by adding constant
brighter_image = original_image + brightness_value

# Example: Add 50 to all pixels
new_image = image + 50
```

#### Contrast Adjustment
```python
# Increase contrast by multiplication
higher_contrast = original_image * contrast_factor

# Example: 1.5x contrast
new_image = image * 1.5
```

#### Image Inversion (Negative)
```python
# For 8-bit images (0-255 range)
negative_image = 255 - original_image

# For normalized images (0.0-1.0 range)
negative_image = 1.0 - original_image
```

### 2. Thresholding
Convert grayscale to binary (black/white):
```python
# Simple thresholding
threshold = 128
binary_image = (image > threshold) * 255
```

### 3. Clipping/Clamping
Ensure values stay within valid range:
```python
# Clip values to 0-255 range
clipped_image = np.clip(image, 0, 255)
```

---

## Code Examples

### Basic Image Loading and Manipulation
```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Load image using different libraries
# Using PIL
pil_image = Image.open('image.jpg')
image_array = np.array(pil_image)

# Using OpenCV
cv_image = cv2.imread('image.jpg')  # BGR format
cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

# Using matplotlib
mpl_image = plt.imread('image.jpg')
```

### Creating Synthetic Images
```python
# Create a grayscale gradient
height, width = 100, 100
gradient = np.zeros((height, width))
for i in range(height):
    gradient[i, :] = i * (255 / height)

# Create RGB color blocks
rgb_image = np.zeros((100, 150, 3), dtype=np.uint8)
rgb_image[:, 0:50, 0] = 255      # Red block
rgb_image[:, 50:100, 1] = 255    # Green block  
rgb_image[:, 100:150, 2] = 255   # Blue block
```

### Basic Operations
```python
# Brightness and contrast
def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    adjusted = image.astype(np.float32)
    adjusted = adjusted * contrast + brightness
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)

# Convert RGB to Grayscale
def rgb_to_grayscale(rgb_image):
    # Weighted average (luminance formula)
    return 0.299 * rgb_image[:,:,0] + 0.587 * rgb_image[:,:,1] + 0.114 * rgb_image[:,:,2]

# Image statistics
def image_stats(image):
    return {
        'shape': image.shape,
        'dtype': image.dtype,
        'min': np.min(image),
        'max': np.max(image),
        'mean': np.mean(image),
        'std': np.std(image)
    }
```

### Display Images
```python
# Display single image
plt.figure(figsize=(8, 6))
plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
plt.title('Image Title')
plt.axis('off')
plt.show()

# Display multiple images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(processed_image, cmap='gray')
axes[1].set_title('Processed')
axes[2].imshow(result_image, cmap='gray')
axes[2].set_title('Result')
for ax in axes:
    ax.axis('off')
plt.show()
```

---

## Practical Exercises

### Exercise 1: Basic Image Creation
```python
# Create a 10×10 checkerboard pattern
checkerboard = np.zeros((10, 10))
checkerboard[::2, ::2] = 255  # White squares
checkerboard[1::2, 1::2] = 255
```

### Exercise 2: Color Channel Manipulation
```python
# Separate RGB channels
red_channel = rgb_image[:, :, 0]
green_channel = rgb_image[:, :, 1]  
blue_channel = rgb_image[:, :, 2]

# Create channel-only images
red_only = rgb_image.copy()
red_only[:, :, 1:3] = 0  # Zero out green and blue
```

### Exercise 3: Image Transformations
```python
# Create simple transformations
def apply_sepia(image):
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])
    return np.dot(image, sepia_filter.T)

def increase_saturation(image, factor=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
```

---

## Key Takeaways

### Fundamental Concepts
1. **Images are numerical arrays** - Everything in computer vision starts with numbers
2. **Spatial information matters** - Pixel positions and neighborhoods are crucial
3. **Multiple representations exist** - Grayscale, RGB, HSV, etc., each useful for different tasks
4. **Preprocessing is essential** - Raw pixels often need adjustment before analysis

### Best Practices
- Always check image data type and range (0-255 vs 0.0-1.0)
- Handle edge cases (values outside valid range)
- Understand coordinate systems for your library
- Visualize intermediate results during processing

### Next Steps
- Image filtering and convolution
- Feature detection and extraction
- Histogram analysis and equalization
- Geometric transformations
- Noise reduction techniques

### Common Pitfalls
- Mixing up coordinate systems (x,y vs row,column)
- Forgetting to handle data type conversions
- Not clipping values after mathematical operations
- Confusing BGR vs RGB color orders in OpenCV

---

## Additional Resources
- [OpenCV Documentation](https://docs.opencv.org/)
- [PIL/Pillow Documentation](https://pillow.readthedocs.io/)
- [NumPy Array Operations](https://numpy.org/doc/stable/user/quickstart.html)
- [Matplotlib Image Tutorial](https://matplotlib.org/stable/tutorials/introductory/images.html)

---

*This document covers the fundamental concepts of digital image representation and basic processing operations. Understanding these concepts is essential before moving on to more advanced computer vision techniques like filtering, feature detection, and machine learning approaches.*