🧠 Open-Vocabulary Object Detection with OWL‑ViT
This project demonstrates open-vocabulary object detection using the OWL‑ViT model from Google.
Unlike traditional object detectors limited to fixed classes, OWL‑ViT can detect any object you describe in text, even if it wasn’t seen during training.
🔍 Features:
Upload any image.
Enter one or multiple text prompts (e.g., temple, umbrella, person).
The model draws bounding boxes around detected objects matching your prompts.
Uses a low detection threshold to catch smaller or less obvious objects.
⚙️ Technologies used:
Transformers (by Hugging Face) for the OWL‑ViT model and processor.
PyTorch for deep learning.
PIL and Matplotlib for drawing and visualization.
Google Colab for running interactively in the browser.
✨ This notebook shows how powerful open-vocabulary models can be, as they aren’t limited to fixed classes like COCO or ImageNet — making them ideal for detecting rare, custom, or domain-specific objects.
🧪 Usage guide
✅ Open the notebook in Google Colab.
✅ Upload any image you’d like to analyze.
✅ Enter comma-separated prompts like:
plaintext
Copy
Edit
umbrella, person, temple
✅ The model will:
Predict bounding boxes for each object matching your prompts.
Display the image with highlighted boxes and scores.
Save the result as detected_result.jpg.


