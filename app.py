from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Load model
with open("signlanguagedetectionmodel48x48.json", "r") as f:
    model = model_from_json(f.read())

model.load_weights("signlanguagedetectionmodel48x48.h5")

labels = [
    'A','B','nothing','C','D','E','F','G','H','I','J','K','L',
    'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
]

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            img = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            processed = preprocess_image(img)
            pred = model.predict(processed)

            prediction = labels[np.argmax(pred)]
            confidence = round(float(np.max(pred)) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
