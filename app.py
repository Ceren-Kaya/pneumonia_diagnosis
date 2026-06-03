import os
import io
import uuid
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from tensorflow.keras.applications.densenet import preprocess_input

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

templates = Jinja2Templates(directory="templates")

MODEL_PATH = "best_model.keras"
os.makedirs("outputs", exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def prepare_image(img: Image.Image, target_size=(224, 224)):
    orig_rgb = img.convert("RGB")
    resized = orig_rgb.resize(target_size)
    arr = np.array(resized).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return orig_rgb, arr


def find_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("No Conv2D layer found.")


def make_gradcam_heatmap(model: tf.keras.Model, x: np.ndarray, conv_layer_name: str):
    conv_layer = model.get_layer(conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(x)
        class_score = preds[:, 0]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = tf.cond(max_val > 0, lambda: heatmap / max_val, lambda: heatmap)

    return heatmap.numpy()


def overlay_heatmap_on_image(orig_rgb: Image.Image, heatmap: np.ndarray, alpha=0.35):
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(
        orig_rgb.size, resample=Image.BILINEAR
    )
    heatmap_arr = np.array(heatmap_img).astype(np.float32) / 255.0

    colormap = cm.get_cmap("jet")
    colored = colormap(heatmap_arr)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored)

    blended = Image.blend(
        orig_rgb.convert("RGBA"),
        colored_img.convert("RGBA"),
        alpha=alpha
    )
    return blended.convert("RGB")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "result": None}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("high_recall"),
    alpha: float = Form(0.35)
):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    orig_rgb, x = prepare_image(img)

    prob = float(model.predict(x, verbose=0).ravel()[0])

    threshold = 0.30 if mode == "high_recall" else 0.50
    pred_label = "PNEUMONIA" if prob >= threshold else "NORMAL"

    last_conv_name = find_last_conv_layer_name(model)
    heatmap = make_gradcam_heatmap(model, x, last_conv_name)
    overlay = overlay_heatmap_on_image(orig_rgb, heatmap, alpha=alpha)

    uid = str(uuid.uuid4())

    uploaded_path = f"outputs/{uid}_uploaded.png"
    gradcam_path = f"outputs/{uid}_gradcam.png"

    orig_rgb.save(uploaded_path)
    overlay.save(gradcam_path)

    result = {
        "probability": round(prob * 100, 2),
        "threshold": threshold,
        "mode": "High Recall" if mode == "high_recall" else "Balanced",
        "prediction": pred_label,
        "uploaded_image": f"/{uploaded_path}",
        "gradcam_image": f"/{gradcam_path}",
        "tradeoff": (
            "Lower threshold increases recall and reduces missed pneumonia cases, but may produce more false positives."
            if mode == "high_recall"
            else
            "Higher threshold is more conservative, improving precision while increasing the risk of false negatives."
        )
    }

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "result": result}
    )

@app.get("/results", response_class=HTMLResponse)
def results_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "request": request,
            "metrics_path": "/outputs/metrics.json",
            "cm_30": "/outputs/cm_thr_0.30.png",
            "cm_50": "/outputs/cm_thr_0.50.png",
            "roc": "/outputs/roc_curve.png"
        }
    )