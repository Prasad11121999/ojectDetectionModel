import base64
import os
import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from torchvision import models, transforms
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

# Create FastAPI app
app = FastAPI()

# Mount static folder for serving static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Directory for saving output images
output_dir = "static/segmented_receipts"
os.makedirs(output_dir, exist_ok=True)

# Load pre-trained Mask R-CNN model
model = models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

# Helper function to convert image to base64
def image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/", include_in_schema=False)
async def upload_image(request: Request, file: UploadFile = File(...), threshold: float = Form(...)):
    # Read file into memory
    content = await file.read()
    image_array = np.asarray(bytearray(content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Failed to load image"}

    # Resize for lower memory usage
    image_resized = cv2.resize(image, (800, 800))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image_rgb).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)

    output_images = []
    masks = prediction[0]['masks']
    scores = prediction[0]['scores']

    # Filter masks by threshold and limit top masks
    top_k = 5
    masks = masks[scores > threshold][:top_k]

    for i, mask in enumerate(masks):
        mask = mask[0].mul(255).byte().cpu().numpy()
        mask = np.uint8(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            cropped_receipt = image_resized[y:y+h, x:x+w]
            base64_image = image_to_base64(cropped_receipt)
            output_images.append(f"data:image/jpeg;base64,{base64_image}")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "output_images": output_images
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
