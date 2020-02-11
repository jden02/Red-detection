import cv2
import matplotlib
matplotlib.use("TkAgg")
#from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np

plt.get_backend()

def show1(image):
    plt.figure(figsize=(15,15))
    plt.imshow(image, interpolation = 'nearest')
    
def show_hsv(hsv):
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    show(rgb)
    
def show_mask(mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')

def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return(img)

def find_biggest_contour(image):
    image = image.copy()
    contours,hierarchy = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in
    contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def circle_countour(image, countour):
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(countour)
    cv2.ellipse(image_with_ellipse, ellipse, (0,255,0), 2)
    return image_with_ellipse

image = cv2.imread('./ferrari.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


max_dimension = max(image.shape)
scale = 700/max_dimension
image = cv2.resize(image, None, fx=scale,fy=scale)
image_blur = cv2.GaussianBlur(image, (7, 7), 0)
image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

# filter by color
min_red = np.array([0, 100, 80])
max_red = np.array([10, 256, 256])
mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

# filter by brightness
min_red = np.array([170, 100, 80])
max_red = np.array([180, 256, 256])
mask2 = cv2.inRange(image_blur_hsv, min_red, max_red)

# Concatenate both the mask for better feature extraction
mask = mask1 + mask2

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

# Extract biggest bounding box
big_contour, red_mask = find_biggest_contour(mask_clean)
# Apply mask
overlay = overlay_mask(red_mask, image)
# Draw bounding box
circled = circle_countour(overlay, big_contour)
show1(circled)

plt.show()