import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_coef, iou
from trainUNet import create_dir

H = 256
W = 256


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  # (H, W, 3)
    x = cv2.resize(x, (W, H))
    # x = cv2.cvtColor(x, cv2.COLOR_BGRA2RGB)
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def read_original(path):
    ori_x = cv2.imread(path, cv2.IMREAD_COLOR)  # (H, W, 3)
    ori_x = cv2.resize(ori_x, (W, H))
    # x = cv2.cvtColor(x, cv2.COLOR_BGRA2RGB)
    return ori_x


def merge(path1, path2):
    image1 = Image.open(path1)
    image2 = Image.open(path2)
    image1_size = image1.size
    merged = Image.new('RGB', (2 * image1_size[0], image1_size[1]), (250, 250, 250))
    merged.paste(image1, (0, 0))
    merged.paste(image2, (image1_size[0], 0))
    # add divider
    d = ImageDraw.Draw(merged)
    top = (256, 256)
    bottom = (256, 0)
    line_color = (255, 255, 255)
    d.line([top, bottom], fill=line_color, width=5)
    merged.save("results/merged_image.jpg", "JPEG")


def execute(test, weights):
    np.random.seed(42)
    tf.random.set_seed(42)
    # Results folder
    create_dir("results")

    # Load the model
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
        model = tf.keras.models.load_model("weights/unet.h5")
        # Load model weights for specific task
        model.load_weights(weights)

    # Prediction
    x = read_image(test)
    y_pred = model.predict(x)[0] > 0.5
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred.astype(np.int32)
    inference = y_pred * 225
    original = read_original(test)

    # Save Results
    cv2.imwrite('results/inference.jpg', inference)
    cv2.imwrite('results/original.jpg', original)
    merge('results/original.jpg', 'results/inference.jpg')
