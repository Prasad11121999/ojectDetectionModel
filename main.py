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
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Helper function to convert image to base64
def image_to_base64(image_path: str):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.get("/",include_in_schema=False, response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/",include_in_schema=False)
async def upload_image(request: Request, file: UploadFile = File(...), threshold: float = Form(...)):
    
    
    image_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)

    with open(image_path, "wb") as f:
        f.write(await file.read())

    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Failed to load image"}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image_rgb).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)

    output_images = []
    masks = prediction[0]['masks']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    masks = masks[scores > threshold]
    for i, mask in enumerate(masks):
        mask = mask[0].mul(255).byte().cpu().numpy()
        mask = np.uint8(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            cropped_receipt = image[y:y+h, x:x+w]
            cropped_image_path = os.path.join(output_dir, f"segmented_receipt_{i + 1}.jpg")
            cv2.imwrite(cropped_image_path, cropped_receipt)
            base64_image = image_to_base64(cropped_image_path)
            output_images.append(f"data:image/jpeg;base64,{base64_image}")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "output_images": output_images
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
