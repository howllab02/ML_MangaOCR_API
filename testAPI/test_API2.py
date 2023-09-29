import requests
import cv2

image_path = "../images/015.jpg"

resp1 = requests.post("http://localhost:5000/detection_and_recognition/",
                      files={"file": open(image_path, 'rb')})

result = resp1.json()["text"]

img = cv2.imread(image_path)

for item in result:
    bbox = item["bbox"]
    print(item["text"])
    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255),
                        thickness=2)

cv2.imwrite("../assets/detection.jpg", img)
cv2.namedWindow("page", cv2.WINDOW_NORMAL)
cv2.resizeWindow("page", 700, 500)
cv2.imshow("page", img)
cv2.waitKey()

