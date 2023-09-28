import numpy as np
import cv2

weight = "./models/YoloMangaDetection/Yolo_model.pt"


def crop_img(img, list_bbox):
    list_img = []
    for bbox in list_bbox:
        crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        list_img.append({"img": crop, "bbox": bbox.tolist()})
    return list_img


def img2text(list_img, model):
    list_text = []
    for img in list_img:
        text = model.predict(img["img"])
        list_text.append({"text": text, "bbox": img["bbox"]})
    return list_text


def to_tensor(image_bytes):
    img_np = np.frombuffer(image_bytes, dtype=np.int8)
    cv_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    return cv_img
