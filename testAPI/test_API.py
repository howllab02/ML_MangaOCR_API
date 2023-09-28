import requests
import cv2

image_path = "../images/015.jpg"


def draw_result(img, list_bbox):
    img = cv2.imread(img)
    i = 0
    for bbox in list_bbox:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255),
                            thickness=2)
    cv2.namedWindow("page", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("page", 500, 700)
    cv2.imshow("page", img)

    for bbox in list_bbox:
        i += 1
        crop_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        resp2 = requests.post("http://localhost:5000/recognition/", files={"file": cv2.imencode('.jpg', crop_img)[1]})
        cv2.imshow("img", crop_img)
        print(resp2.json()["text"])
        cv2.waitKey()


resp1 = requests.post("http://localhost:5000/detection/",
                      files={"file": open(image_path, 'rb')})

bbox_list = resp1.json()["bbox"]
draw_result(image_path, bbox_list)
