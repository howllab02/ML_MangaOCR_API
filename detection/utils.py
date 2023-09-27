import numpy as np
import cv2

weight = "./models/YoloMangaDetection/Yolo_model.pt"


def to_tensor(image_bytes):
    img_np = np.frombuffer(image_bytes, dtype=np.int8)
    cv_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    return cv_img
