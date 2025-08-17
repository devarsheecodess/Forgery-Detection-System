import easyocr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import onnxruntime as ort
from PIL import Image
import numpy as np
import io
from gemini.config import generate
from functions.config import postprocess_output, preprocess_image

app = FastAPI()

session = ort.InferenceSession("model/best.onnx")
input_name = session.get_inputs()[0].name

@app.post("/detect")
async def detect_forgery(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file type")
    try:
        img_bytes = await file.read()
        input_tensor, scale, pad = preprocess_image(img_bytes)
        print(f"[DEBUG] Input tensor shape: {input_tensor.shape}, min: {input_tensor.min()}, max: {input_tensor.max()}")
        outputs = session.run(None, {input_name: input_tensor})
        print(f"[DEBUG] Raw model output type: {type(outputs)}, length: {len(outputs)}")
        if len(outputs) > 0:
            print(f"[DEBUG] Output[0] shape: {outputs[0].shape}")
            print(f"[DEBUG] Output[0] sample values: {outputs[0].flatten()[:10]}")
        # Lower confidence threshold for debugging
        detections = postprocess_output(outputs, scale, pad, conf_threshold=0.01)
        if not detections:
            print("[DEBUG] No detections found. Check model output and preprocessing.")
        return JSONResponse(content={"detections": detections})
    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyse")
async def analyse_file(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file type")
    try:
        reader = easyocr.Reader(['en'])  # English
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        results = reader.readtext(np.array(image))

        extracted_text = " ".join([text for _, text, _ in results])

		# hardcoded as of now
        user_data = {
            "name": "John Doe",
            "age": 30,
            "location": "USA"
        }

        response = generate(user_data, extracted_text)

        return response

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
