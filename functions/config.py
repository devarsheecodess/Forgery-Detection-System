import easyocr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import onnxruntime as ort
from PIL import Image
import numpy as np
import io
import os
from dotenv import load_dotenv
from gemini.config import generate

def letterbox_image(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize image to fit into new_shape while keeping aspect ratio using padding (letterbox).
    Returns the padded image, scale factor and padding offsets.
    """
    width, height = image.size
    new_width, new_height = new_shape

    scale = min(new_width / width, new_height / height)
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    resized_image = image.resize((scaled_width, scaled_height), Image.BILINEAR)
    padded_image = Image.new('RGB', new_shape, color)
    pad_x = (new_width - scaled_width) // 2
    pad_y = (new_height - scaled_height) // 2
    padded_image.paste(resized_image, (pad_x, pad_y))

    return padded_image, scale, (pad_x, pad_y)

def preprocess_image(image_bytes: bytes, img_size=640):
    """
    Preprocess image bytes for YOLOv8 ONNX model:
    - Letterbox resize
    - Normalize pixel values [0,1]
    - HWC to CHW
    - Add batch dimension
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image, scale, pad = letterbox_image(image, (img_size, img_size))

    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC to CHW
    img_np = np.expand_dims(img_np, axis=0)   # batch dim

    return img_np, scale, pad

def scale_coords(coords, scale, pad):
    """
    Scale bounding box coordinates back to original image size.
    coords: list or array [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = coords
    pad_x, pad_y = pad

    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    return [x1, y1, x2, y2]

def postprocess_output(outputs, scale, pad, conf_threshold=0.25):
    """
    Postprocess YOLOv8 ONNX model output.
    - output shape expected: [1, 85, N]
    - where 85 = 4 bbox coords + 1 objectness + 80 class scores (for COCO)
    - filters by confidence threshold (objectness * class probability)
    - converts bbox format center_x,y,w,h to x1,y1,x2,y2
    - scales coordinates back to original image size
    """
    detections = []

    output = outputs[0]
    # Handle (1, 85, N), (1, N, 85), and (1, 5, N) shapes
    if output.shape[1] == 85:
        # (1, 85, N) -> (N, 85)
        output = output.transpose(0, 2, 1)[0]
        multi_class = True
    elif output.shape[2] == 85:
        # (1, N, 85) -> (N, 85)
        output = output[0]
        multi_class = True
    elif output.shape[1] == 5:
        # (1, 5, N) -> (N, 5), single-class, no class probabilities
        output = output.transpose(0, 2, 1)[0]
        multi_class = False
    else:
        print(f"[DEBUG] Unexpected output shape: {output.shape}")
        return []


    for det in output:
        if multi_class:
            obj_conf = det[4]
            class_probs = det[5:]
            if class_probs.size == 0:
                continue
            class_id = int(np.argmax(class_probs))
            class_conf = class_probs[class_id]
            conf = obj_conf * class_conf
        else:
            # Only 5 values: x, y, w, h, objectness
            obj_conf = det[4]
            class_id = 0
            conf = obj_conf
        if conf >= conf_threshold:
            x_c, y_c, w, h = det[:4]
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            x2 = x_c + w / 2
            y2 = y_c + h / 2
            bbox = scale_coords([x1, y1, x2, y2], scale, pad)
            detections.append({
                "bbox": [float(coord) for coord in bbox],
                "confidence": float(conf),
                "class_id": class_id
            })

    # Debug: print number of detections and first few raw outputs
    print(f"[DEBUG] Num detections: {len(detections)}")
    print(f"[DEBUG] First 3 output rows: {output[:3]}")

    return detections
