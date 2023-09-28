import torch
import supervision as sv


class YoloModel:
    def __init__(self, weight):
        self.weight = weight
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', weight)

    def predict(self, np_image):
        results = self.model(np_image)
        detections = sv.Detections.from_yolov5(results).with_nms(threshold=0.1)

        return detections.xyxy
