import cv2
import csv
from os.path import exists
import pytesseract
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# point to license plate image (works well with custom crop function)
image = cv2.imread("./croppedimages/8.png", 0)

# gray = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
# Perform histogram equalization
# gray = cv2.equalizeHist(gray)

blur = cv2.GaussianBlur(image, (5, 5), 0)
gray = cv2.medianBlur(image, 3)
# perform otsu thresh (using binary inverse since opencv contours work better with white text)
ret, thresh = cv2.threshold(
    blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
cv2.imshow("Otsu", thresh)
cv2.waitKey(0)
rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# apply dilation
dilation = cv2.dilate(thresh, rect_kern, iterations=1)
cv2.imshow("dilation", dilation)
cv2.waitKey(0)


opening = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN, rect_kern, iterations=2)
cv2.imshow("opening", opening)
cv2.waitKey(0)


# find contours
try:
    contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
except:
    ret_img, contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

# create copy of image
im2 = gray.copy()

plate_num = []
# loop through contours and find letters in license plate
for cnt in sorted_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    height, width = im2.shape

    # if height of box is not a quarter of total height then skip
    if height / float(h) > 6:
        continue
    ratio = h / float(w)
    # if height to width ratio is less than 1.5 skip
    if ratio < 1.5:
        continue
    area = h * w
    # if width is not more than 25 pixels skip
    if width / float(w) > 15:
        continue
    # if area is less than 100 pixels skip
    if area < 100:
        continue
    # draw the rectangle
    rect = cv2.rectangle(im2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi = thresh[y-5:y+h+5, x-5:x+w+5]
    roi = cv2.bitwise_not(roi)
    roi = cv2.medianBlur(roi, 5)
    # cv2.imshow("ROI", roi)
    # cv2.waitKey(0)
    text = pytesseract.image_to_string(
        roi, config='-l eng -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3')
    # text = pytesseract.image_to_string(
    # roi, config='-l eng --oem 1 --psm 3')
    plate_num.append(text.strip())

cv2.imshow("Character's Segmented", im2)
cv2.waitKey(0)
print("Vehicle License Plate: " + ''.join(plate_num))

# Get current datetime
now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y")
t_string = now.strftime("%H:%M:%S")

print("date and time =", dt_string)

fields = ['License Plate Number', 'Vehicle Type', 'Date', 'Time']
data = [[''.join(plate_num), 'Car', dt_string, t_string]]

# Write to CSV file
if exists("./lp_info.csv"):
    with open('lp_info.csv', 'a', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the data rows
        csvwriter.writerows(data)
else:
    with open('lp_info.csv', 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerows(data)

cv2.destroyAllWindows()
