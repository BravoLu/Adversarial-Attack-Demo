import os
import cv2
import numpy as np
#import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import ResNet18
import uuid
from flask import Flask, render_template, request, flash, redirect, session

app = Flask(__name__)
app.secret_key = "super secret key"

def normalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    for c in range(3):
        tensor.data[:, c, :, :] = (tensor.data[:, c, :, :] - mean[c]) / std[c]
    return tensor


def classifier(img_name, load_folder, model_path):
    weight = torch.load(model_path, map_location=torch.device('cpu'))
    model = ResNet18()
    model.load_state_dict(weight)
    model.eval()

    load_path = os.path.join(load_folder, img_name)

    image = cv2.imread(load_path)
    image = cv2.resize(image, (224, 224))
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image).unsqueeze(0)
    image = image / 255.0
    image = normalize(image)
    output = model(image.float())
    possibility = F.softmax(output, 1)
    _, label = output.max(1)
    return possibility.detach().numpy().tolist(), label

# create folder for Uploading and cartoonizing images
model_path = "saved_models/resnet18.pth"
UPLOAD_FOLDER = "static/upload/"
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])


def allowed_files(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index_toonit.html")


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        # read the POST request input
        if "file" not in request.files:
            flash("No file part")
        raw_image, adv_image = request.files["raw_image"], request.files["adv_image"]
        if raw_image and allowed_files(raw_image.filename):
            print(raw_image.filename)
            extension = os.path.splitext(raw_image.filename)[1]
            raw_img_name = str(uuid.uuid4()) + extension

            raw_image.save(os.path.join(UPLOAD_FOLDER, raw_img_name))
            image_location = os.path.join(UPLOAD_FOLDER, raw_img_name)

            # pass hashed values
            raw_pos, raw_label = classifier(raw_img_name, UPLOAD_FOLDER, model_path)
            raw_pos = [float('{:.2f}'.format(i)) for i in raw_pos[0]]
        else:
            flash("Allowed image types are -> png, jpg, jpeg, gif")
            return redirect(request.url)

        if adv_image and allowed_files(adv_image.filename):
            print(adv_image.filename)
            extension = os.path.splitext(adv_image.filename)[1]
            adv_img_name = str(uuid.uuid4()) + extension

            adv_image.save(os.path.join(UPLOAD_FOLDER, adv_img_name))
            image_location = os.path.join(UPLOAD_FOLDER, adv_img_name)

            # pass hashed values
            adv_pos, adv_label = classifier(adv_img_name, UPLOAD_FOLDER, model_path)
            print(adv_pos[0])
            adv_pos = [float('{:.2f}'.format(i)) for i in adv_pos[0]]
        else:
            flash("Allowed image types are -> png, jpg, jpeg, gif")
            return redirect(request.url)

        return render_template("result.html", raw=raw_img_name, adv=adv_img_name,  raw_pos=raw_pos, adv_pos=adv_pos)

@app.route("/hello/<name>")
def hello(name):
    return "hello %s!" % name


if __name__ == "__main__":
    app.run(host='0.0.0.0', port="9999", debug=True)
