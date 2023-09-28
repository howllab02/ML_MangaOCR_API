from flask import Flask, jsonify, request
from detection.detection_model import YoloModel
from recognition.rec_model import OCRModel
from functions.utils import crop_img, img2text, to_tensor, weight

app = Flask(__name__)

model_detection = YoloModel(weight)
model_rec = OCRModel("./models/TrOCRMangaRec/best_model", "./recognition/tokenizer", "./recognition/processor")


@app.route('/', methods=["GET"])
def hello_world():  # put application's code here
    return 'Welcome'


@app.route('/detection/', methods=["POST"])
def detection():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        np_img = to_tensor(img_bytes)
        result = model_detection.predict(np_image=np_img)
        return jsonify({"bbox": result.tolist()})


@app.route("/recognition/", methods=["POST"])
def recognition():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        np_img = to_tensor(img_bytes)
        result = model_rec.predict(np_img)
        return jsonify({"text": result})


@app.route("/detection_and_recognition/", methods=["POST"])
def detect_and_recognition():
    if request.method == 'POST':
        file = request.files['file']
        image_bytes = file.read()
        np_img = to_tensor(image_bytes)
        bbox = model_detection.predict(np_img)
        text_img = crop_img(np_img, bbox)
        list_text = img2text(list_img=text_img, model=model_rec)
        return jsonify({"text": list_text})


if __name__ == '__main__':
    app.run()
