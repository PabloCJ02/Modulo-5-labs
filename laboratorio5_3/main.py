import pytesseract
import matplotlib.pyplot as plt
import cv2
import glob
import os

test_license_plate = cv2.imread(os.getcwd() + "/license-plates/JSQ1413.jpg")
plt.imshow(test_license_plate)
plt.axis('off')
plt.title('JSQ1413 license plate')

resize_test_license_plate_JSQ1413 = cv2.resize(test_license_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
grayscale_resize_test_license_plate_JSQ1413 = cv2.cvtColor(resize_test_license_plate_JSQ1413, cv2.COLOR_BGR2GRAY)
gaussian_blur_license_plate_JSQ1413 = cv2.GaussianBlur(grayscale_resize_test_license_plate_JSQ1413, (5, 5), 0)

new_predicted_result_JSQ1413 = pytesseract.image_to_string(gaussian_blur_license_plate_JSQ1413, lang='eng',
config='--oem 3 -l eng --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
filter_new_predicted_result_JSQ1413 = "".join(new_predicted_result_JSQ1413.split()).replace(":", "").replace("-", "")
print(filter_new_predicted_result_JSQ1413)
