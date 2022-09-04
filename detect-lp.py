import math
from turtle import width
import cv2
import imutils
import deskew as ds

# Read the image
image = cv2.imread("./samples/car6.PNG")

# Resize to width 500
image = imutils.resize(image, width=500)

# Show original image
cv2.imshow("Original image", image)
cv2.waitKey(0)

# Convert image from RGB to Gray Scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scaled image", gray_image)
cv2.waitKey(0)

# Bilateral filter removes noise while keeping edges sharp
blurred_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
cv2.imshow("Smoothened image (Bilateral Filter)", blurred_image)
cv2.waitKey(0)

# Find edges using canny detector
edged_image = cv2.Canny(blurred_image, 170, 200)
cv2.imshow("Canny Edged image", edged_image)
cv2.waitKey(0)

# Find contours based on edges
cnts, new = cv2.findContours(
    edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Create copy of original image to draw all contours
image_new = image.copy()
cv2.drawContours(image_new, cnts, -1, (0, 255, 0), 3)
cv2.imshow("All Contours", image_new)
cv2.waitKey(0)

# sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

# Top 30 Contours
image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Top 30 contours", image2)
cv2.waitKey(0)

NumberPlateCnt = None  # Number plate countour init

# loop over our contours to find the best possible approximate contour of number plate
i = 7
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)

    if len(approx) == 4:  # Select the contour with 4 corners
        NumberPlateCnt = approx  # This is our approx Number Plate Contour
        break

if NumberPlateCnt is None:
    lp_detected = False
    print("Couldn't detect LP (No contours)")
else:
    lp_detected = True  # LP detected

if lp_detected == True:
    """
    Crop those contours and store it in Cropped Images folder
    """
    # This will find out coordinates for plate
    x, y, w, h = cv2.boundingRect(c)
    # Create new image
    plate_image = image[y:y+h, x:x+w]

    # Store new image
    cv2.imwrite('C:\\Users\\LENOVO\\Documents\\Projects\\AutoVIS\\croppedimages\\' +
                str(i) + '.png', plate_image)

    # Drawing the selected contours on the original image
    cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
    cv2.imshow("Image with detected license plate", image)
    cv2.waitKey(0)

# Show cropped LP
cropped_img = cv2.imread('./croppedimages/7.png')
cv2.imshow("Cropped License Plate", cropped_img)
cv2.waitKey(0)

# Deskew the cropped LP
corrected_img, angle = ds.deskew(cropped_img)
cv2.imshow("Corrected image", corrected_img)
cv2.waitKey(0)
print(angle)

#corrected_img = imutils.resize(corrected_img, width=500, height=200)
corrected_img = cv2.resize(corrected_img, None, fx=3,
                           fy=3, interpolation=cv2.INTER_CUBIC)

print("Shape of the image", corrected_img.shape)

h, w, _ = corrected_img.shape

height_pixels = h
width_pixels = w
crop_pixels = math.ceil(height_pixels/7)

# Crop till clear license plate
if -8 <= angle <= -4:
    crop_pixels = math.ceil(height_pixels/7)
    crop = corrected_img[crop_pixels:height_pixels-crop_pixels, ]
    cv2.imshow("LOOOOOOOO", crop)
    cv2.waitKey(0)

if angle <= -8:
    crop_pixels = math.ceil(height_pixels/5)
    crop = corrected_img[crop_pixels:height_pixels-crop_pixels, ]

    cv2.imshow("LOOOOOOOO", crop)
    cv2.waitKey(0)

cv2.imwrite('C:\\Users\\LENOVO\\Documents\\Projects\\AutoVIS\\croppedimages\\'
            + '8.png', corrected_img)

cv2.destroyAllWindows()

"""Another way of cropping
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]
"""
