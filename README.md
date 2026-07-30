# 🛰️ Multispectral Water Bodies Segmentation using DeepLabV3+

An end-to-end Deep Learning pipeline designed to segment water bodies from 12-channel multispectral satellite imagery (Sentinel-2) using state-of-the-art semantic segmentation architecture.

---

## 👤 Author

* **Tala Rashad** - [GitHub Profile](https://github.com/talarashad)

---

## 🌟 Key Features

* **Multi-Spectral Processing**: Seamless handling of 12-band raster inputs using `rasterio`.
* **Data Leakage Prevention**: Per-channel global min/max stats computed strictly on the training split.
* **Synchronized Data Augmentations**: Joint spatial augmentations (Flips & Rotations) applied synchronously to images and masks.
* **Loss Function Engineering**: Combined Loss (`BCEWithLogitsLoss` + `DiceLoss`) for robust gradient stability and overlap optimization.
* **Advanced Architecture**: Fine-tuning pretrained `DeepLabV3+` with a `ResNet-34` backbone.
* **Interactive Web Deployment**: Integrated **Flask Web Application** to upload satellite images and generate real-time segmentation masks.
---
## 📁 Dataset

The dataset used in this project consists of 12-band Sentinel-2 multispectral satellite imagery for water body segmentation. 
You can access and download the dataset here: [Download Dataset](https://drive.google.com/drive/folders/1GQss5oZhv-0dxoRtI5m_m_SPbm-E3vXJ?usp=drive_link)
## 📊 Model Evaluation & Performance Metrics

Quantitative evaluation results of the fine-tuned **DeepLabV3+ (ResNet-34)** model on the validation dataset:

| Metric | DeepLabV3+ (ResNet-34 Pretrained) |
| :--- | :---: |
| **IoU (Intersection over Union)** | **`77.60%`** 🚀 |
| **F1-Score (Dice)** | **`87.26%`** |
| **Precision** | **`87.92%`** |
| **Recall / Sensitivity** | **`86.61%`** |

---

## 🏗️ Repository Structure

```text
├── inference_samples/              # Original satellite test images for model inference
├── sample_test_images/             # Screenshots and visual results from app testing
├── templates/                      # HTML templates for Flask front-end
├── app.py                          # Flask web deployment script
├── train_model.py                  # AI Model training & evaluation pipeline
├── best_deeplabv3plus_model.pth    # Fine-tuned model weights
└── requirements.txt                # Project dependencies

##🧠 Training Pipeline
├── 1. Google Drive Mounting & Dataset Alignment (.tif / .png matching)
├── 2. Channel-Wise Vectorized Global Min-Max Normalization
├── 3. Custom PyTorch WaterDataset Pipeline
├── 4. DeepLabV3+ (ResNet-34 Encoder) Initialization
├── 5. BCE + Dice Loss Formulation & AdamW Optimization
├── 6. Training Loop with ReduceLROnPlateau Scheduler
└── 7. Performance Evaluation & Dynamic RGB Stretch Visualization

##⚙️ Hyperparameters

Input Channels : 12 Bands
Image Resolution: 128 x 128
Batch Size       : 16
Optimizer        : AdamW (LR = 3e-4, Weight Decay = 1e-2)
Scheduler        : ReduceLROnPlateau (Factor = 0.5, Patience = 3)
Loss Function    : BCEWithLogitsLoss + DiceLoss
Epochs           : 25

##🛠️Installation & Setup
1.Clone the Repository:
Bash
git clone [https://github.com/talarashad/Satellite-Water-Segmentation-DeepLabV3Plus.git](https://github.com/talarashad/Satellite-Water-Segmentation-DeepLabV3Plus.git)
cd Satellite-Water-Segmentation-DeepLabV3Plus

2.Install Requirements:
pip install torch torchvision rasterio segmentation-models-pytorch scikit-learn matplotlib Pillow

3.Run Notebook:
Open the main notebook in Google Colab or Jupyter and execute cells sequentially (Cell 1 to Cell 10).

##🌐 Flask Web Application
A Flask web application is provided to deploy the trained segmentation model locally:
1.Run the Application:
Bash
python app.py

2.Access the Interface:
Open your browser and navigate to http://127.0.0.1:5000.

3.Test with Samples:
Upload any satellite image from the inference_samples/ directory to generate and view predicted water segmentation masks in real time.
