from flask import Flask, render_template, request
import os
import cv2
from main import run_pipeline

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static/results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html")

        upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_path)

        # Run cnn-segmentation pipeline
        original_image, result_image = run_pipeline(upload_path)

        # Save only boundary result
        result_filename = "result_" + file.filename
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, result_image)

        # Overwrite uploaded image with resized version
        cv2.imwrite(upload_path, original_image)

        return render_template(
            "index.html",
            original=file.filename,
            result=result_filename
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
