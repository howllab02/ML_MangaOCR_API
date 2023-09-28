from flask import Flask, jsonify, request
from detection.utils import weight, to_tensor
from detection.detection_model import YoloModel
from recognition.rec_model import OCRModel

app = Flask(__name__)


@app.route('/', methods=["GET"])
def hello_world():  # put application's code here
    return 'Welcome'


@app.route('/detection/', methods=["POST"])
def detection():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        np_img = to_tensor(img_bytes)
        model = YoloModel(weight)
        result = model.predict(np_img)
        return jsonify({"bbox": result.tolist()})


@app.route("/recognition/", methods=["POST"])
def recognition():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        np_img = to_tensor(img_bytes)
        model = OCRModel("./models/TrOCRMangaRec/best_model","./recognition/tokenizer", "./recognition/processor")
        result = model.predict(np_img)
        return jsonify({"text": result})


if __name__ == '__main__':
    app.run()
