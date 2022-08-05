import cv2
import imutils

# Read the image
image = cv2.imread("./samples/image16.PNG")

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
gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
cv2.imshow("Smoothened image (Bilateral Filter)", gray_image)
cv2.waitKey(0)

# Find edges using canny detector
edged_image = cv2.Canny(gray_image, 170, 200)
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

NumberPlateCnt = None  # we currently have no Number plate contour

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
    Cropped_loc = './croppedimages/7.png'
    cv2.imshow("Cropped License Plate", cv2.imread(Cropped_loc))
    cv2.waitKey(0)
cv2.destroyAllWindows()
