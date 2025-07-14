!pip install -q transformers torch torchvision pillow matplotlib

import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()
image_path = next(iter(uploaded))
image = Image.open(image_path).convert("RGB")

prompts = input("Enter comma-separated object prompts: ").split(',')
prompts = [p.strip() for p in prompts if p.strip()]

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

inputs = processor(text=[prompts], images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

width, height = image.size
target_sizes = torch.tensor([[height, width]])
results = processor.post_process_grounded_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.02
)[0]

if results['boxes'].shape[0] > 0:
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x0, y0, x1, y1 = box.tolist()
        prompt_label = prompts[label]
        draw.rectangle([x0, y0, x1, y1], outline="yellow", width=4)
        draw.text((x0, max(y0-10, 0)), f"{prompt_label}: {score:.2f}", fill="black")

    plt.figure(figsize=(10,10))
    plt.imshow(image_draw)
    plt.axis('off')
    plt.show()

    image_draw.save("detected_result.jpg")
    print("Detection complete. Saved as detected_result.jpg")
else:
    print("⚠️ No objects detected for the given prompts.")