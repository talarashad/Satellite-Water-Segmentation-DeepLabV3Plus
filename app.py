import io
import torch
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image

app = Flask(__name__)

# 1. Device configuration and model path definition
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "best_deeplabv3plus_model.pth"

# 2. Global Min and Max values computed per channel from the training set
GLOBAL_MIN = np.array([-1393.0, -1169.0, -722.0, -684.0, -412.0, -335.0, -251.0, 64.0, -9999.0, 8.0, 10.0, 0.0], dtype=np.float32)
GLOBAL_MAX = np.array([6568.0, 9659.0, 11368.0, 12041.0, 15841.0, 15252.0, 14647.0, 255.0, 4245.0, 4287.0, 100.0, 111.0], dtype=np.float32)

# Pre-compute reshaped tensors and denominator to avoid redundant calculations
GLOBAL_MIN_RESHAPED = GLOBAL_MIN.reshape(-1, 1, 1).astype(np.float32)
DENOM = np.maximum((GLOBAL_MAX - GLOBAL_MIN).reshape(-1, 1, 1).astype(np.float32), 1e-8)

# 3. Instantiate the DeepLabV3+ model architecture
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=12,
    classes=1
)

# 4. Load trained model weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights from '{MODEL_PATH}': {e}")

model.to(device)
model.eval()

def preprocess_12ch_image(file_bytes):
    """
    Reads a 12-channel multispectral raster image from memory bytes
    and applies global Min-Max normalization consistent with training.
    """
    with rasterio.MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            image = src.read().astype(np.float32)  # Shape: (12, H, W)
            
            # Apply Min-Max normalization and clip values in-place
            image = (image - GLOBAL_MIN_RESHAPED) / DENOM
            np.clip(image, 0.0, 1.0, out=image)
            
            # Convert NumPy array to PyTorch Tensor and add batch dimension -> (1, 12, H, W)
            tensor = torch.from_numpy(image).unsqueeze(0)
            return tensor.to(device)

@app.route('/')
def home():
    """Renders the HTML user interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to receive an image file and return the predicted binary mask."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_bytes = file.read()
        input_tensor = preprocess_12ch_image(file_bytes)

        # Run inference without tracking gradients
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output) > 0.5
            pred_mask = pred.squeeze().cpu().numpy().astype(np.uint8) * 255

        # Convert the binary mask array to PIL Image and prepare response
        mask_pil = Image.fromarray(pred_mask)
        img_io = io.BytesIO()
        mask_pil.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)