import json
import logging
from typing import List

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image
from tqdm import trange

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

CLIP_MODEL = "ViT-bigG-14"
CLIP_DATASET = "laion2b_s39b_b160k"
BATCH_SIZE = 10
DEVICE = "cuda"

(
    model,
    _,
    preprocess,
) = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_DATASET)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer(CLIP_MODEL)


def get_image_embeddings(image_paths: List[str]) -> List[List[float]]:
    for i in trange(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[i : i + BATCH_SIZE]
        if len(batch) == 0:
            continue
        images = torch.stack(
            [preprocess(Image.open(img).convert("RGB")) for img in batch]
        ).to(DEVICE)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            image_features = image_features.cpu().numpy()
            if i == 0:
                embeddings = image_features
            else:
                embeddings = np.concatenate((embeddings, image_features))
    return embeddings.tolist()


def get_text_embeddings(texts: List[str]) -> List[List[float]]:
    for i in trange(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        text = tokenizer(batch).to(DEVICE)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
            text_features = text_features.cpu().numpy()
            if i == 0:
                embeddings = text_features
            else:
                embeddings = np.concatenate((embeddings, text_features))
    return embeddings.tolist()


@app.route("/", methods=["POST"])
def interact_with_clip():
    logging.info(request.form)
    if "image" in request.form:
        images = json.loads(request.form["image"])
        logging.info(images)
        embeddings = get_image_embeddings(images)

    if "text" in request.form:
        texts = json.loads(request.form["text"])
        logging.info(texts)
        embeddings = get_text_embeddings(texts)

    return jsonify({"embeddings": embeddings})


if __name__ == "__main__":
    logging.info("Server is running!")
    app.run(host="0.0.0.0", port=8090, debug=False)
