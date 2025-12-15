import os
import io
import json
import shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import torch
import timm
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# ---------- PATH ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR 
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
PLOTS_DIR = os.path.join(STATIC_DIR, "plots")
METRICS_PLOTS_DIR = os.path.join(STATIC_DIR, "plots_metrics")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_PLOTS_DIR, exist_ok=True)


# ---------- CONFIG ----------
MODEL_PATH = os.path.join(ROOT_DIR, "model", "best_model.pth")
CLASS_PATH = os.path.join(ROOT_DIR, "model", "class_names.json")
HISTORY_PATH = os.path.join(ROOT_DIR, "model", "history_swin.json")
TEST_DATA_DIR = os.path.join(ROOT_DIR, "dataset", "test")

DEVICE = torch.device("cpu") 


# ---------- APP ----------
app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    template_folder=TEMPLATE_DIR
)
app.secret_key = "change-this-secret"


# FULL CLASS NAMES
CLASS_FULL_NAMES = {
    "akiec": "Actinic keratoses / Intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi (common mole)",
    "vasc": "Vascular lesions (angiomas, etc.)"
}

# ---------- MODEL LOAD ----------
print("Loading class names...")
try:
    with open(CLASS_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
except FileNotFoundError:
    CLASS_NAMES = ["class1", "class2"]

NUM_CLASSES = len(CLASS_NAMES)

print("Loading model...")

MODEL_NAME = "swin_tiny_patch4_window7_224"
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

if os.path.exists(MODEL_PATH):

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    print("Model weights loaded.")
else:
    print(f"Warning: {MODEL_PATH} not found.")

model.to(DEVICE)
model.eval()


# ---------- TRANSFORM ----------
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])


# ---------- HELPERS ----------
def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            pass 

def predict_image_pil(pil_image):
    img = pil_image.convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_short = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
    return pred_short, confidence, probs

def save_plot(fig, filename, folder_path):

    import matplotlib.pyplot as plt
    path = os.path.join(folder_path, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return filename

def plot_history(history_path=HISTORY_PATH):
    if not os.path.exists(history_path):
        return None
    

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    clear_folder(PLOTS_DIR)
    
    with open(history_path, "r") as f:
        h = json.load(f)

    train_loss = h.get("train_loss", [])
    val_loss = h.get("val_loss", [])
    train_acc = h.get("train_acc", [])
    val_acc = h.get("val_acc", [])

    ts = datetime.now().strftime('%Y%m%d%H%M%S')

    fig1 = plt.figure(figsize=(8, 4))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    loss_path = save_plot(fig1, f"loss_{ts}.png",PLOTS_DIR)

    fig2 = plt.figure(figsize=(8, 4))
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend()
    acc_path = save_plot(fig2, f"acc_{ts}.png",PLOTS_DIR)

    return loss_path, acc_path

def compute_confusion_and_report(test_dir=TEST_DATA_DIR, max_images_per_class=None):
    y_true = []
    y_pred = []
    if not os.path.isdir(test_dir):
        return None, None, "No test dataset folder found."

    for cls in sorted(os.listdir(test_dir)):
        cls_folder = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_folder):
            continue
        files = [f for f in os.listdir(cls_folder) if f.lower().endswith((".jpg", ".png"))]
        if max_images_per_class:
            files = files[:max_images_per_class]

        for p in files:
            try:
                pil = Image.open(os.path.join(cls_folder, p)).convert("RGB")
                pred_short, _, _ = predict_image_pil(pil)
                y_true.append(cls)
                y_pred.append(pred_short)
            except:
                pass

    if len(y_true) == 0:
        return None, None, "No images found."

    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=labels)
    return (cm, labels, report)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    if cm is None: return None
    

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    path = save_plot(fig, f"cm_{title}_{ts}.png", METRICS_PLOTS_DIR)
    return path

# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html", classes=CLASS_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files: return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "": return redirect(url_for("index"))

    img = Image.open(file.stream).convert("RGB")
    pred_short, confidence, _ = predict_image_pil(img)
    pred_full = CLASS_FULL_NAMES.get(pred_short, pred_short)

    clear_folder(UPLOAD_DIR)
    filename = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    img.save(os.path.join(UPLOAD_DIR, filename))


    loss_plot, acc_plot = None, None
    try:
        plots = plot_history()
        if plots: loss_plot, acc_plot = plots
    except Exception as e:
        print("Plotting error (skipped to save memory):", e)

    return render_template(
        "result.html",
        image_url="/static/uploads/" + filename,
        pred=pred_full,
        pred_short=pred_short,
        confidence=f"{confidence*100:.2f}%",
        classes=CLASS_NAMES,
        loss_plot=loss_plot,
        acc_plot=acc_plot
    )

@app.route("/metrics")
def metrics():
    clear_folder(METRICS_PLOTS_DIR)

    cm, labels, report = compute_confusion_and_report(max_images_per_class=20)
    
    if isinstance(report, str) and cm is None:
        return f"<h3>{report}</h3>"

    cm_path = plot_confusion_matrix(cm, labels, normalize=False, title="Confusion")
    cm_norm_path = plot_confusion_matrix(cm, labels, normalize=True, title="Norm_Confusion")
    return render_template("metrics.html", cm_path=cm_path, cm_norm_path=cm_norm_path, report=report)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)