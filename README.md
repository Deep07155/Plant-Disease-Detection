# 🌿 LeafLens — Plant Disease Detection

A premium, interactive **Streamlit + PyTorch** plant disease detection website using a **real, already-trained MobileNetV2 model**. Includes three design options inspired by the approved concept previews: **Premium Dark (default)**, Natural & Clean, and AgriTech Dashboard.

## Features

- Upload a leaf photo (JPG/PNG) and receive an actual prediction.
- View the predicted plant and condition, model confidence, and the top three categories.
- Browse original training/validation **accuracy and loss** graphs.
- Switch between three visual styles. All layouts use the **same trained model**.
- Download a text prediction report and read general next-step advice.
- No API key, FastAPI server, or model retraining required.

## Project structure

```text
Plant_Disease_Streamlit_Website/
├── app.py                       # Website, themes, pages, upload controls
├── predict.py                   # Load MobileNetV2 + image prediction
├── model/
│   ├── mobilenet_best.pth       # Your trained MobileNetV2 weights (included)
│   └── class_names.json        # Your exact 38 saved class names (included)
├── image/
│   ├── accuracy_comparison.png # Your notebook's accuracy graph
│   └── loss_comparison.png     # Your notebook's loss graph
├── assets/
│   └── hero.jpg                # Website hero photo
├── notebooks/plant-disease-detection.ipynb  # Your final Kaggle notebook
├── .streamlit/config.toml
├── requirements.txt
└── .gitignore
```

## Run on your computer

Recommended: **Python 3.11 or 3.12**.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell prevents activation, use the terminal commands directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

1. Upload this entire project to a GitHub repository; include the `model/mobilenet_best.pth` file (about 9 MB) and `model/class_names.json`.
2. Open [share.streamlit.io](https://share.streamlit.io/), create an app, and connect the repository.
3. Choose **`app.py`** as the entrypoint. Select a compatible Python version (3.11 or 3.12) in advanced settings if the platform offers that option.
4. Deploy. The model will load on CPU; GPU is not required.

If torch installation is too large for your hosting environment, consider a deployment platform with more memory. Keep the image size and normalization unchanged to match the model you trained.

## Your verified training results

| Saved model | Validation accuracy | Saved epoch |
| --- | ---: | ---: |
| Custom CNN | 95.56% | 14 |
| **MobileNetV2 (deployed)** | **96.53%** | **8** |

These numbers come from your uploaded Kaggle notebook and reflect the model checkpoints selected by lowest validation loss. They are **validation scores**, not independent field or test results. The supplied dataset already contains augmented images, so real-world performance may vary.

## How prediction works

1. `Image.open()` reads the user-uploaded leaf photo.
2. `transforms.Resize((160,160))` resizes the picture to the exact training input size.
3. `ToTensor()` and `Normalize(...)` apply the same preprocessing as training.
4. `torchvision.models.mobilenet_v2(weights=None)` constructs the model without downloading pretrained weights.
5. `load_state_dict()` loads **your `mobilenet_best.pth`** exactly.
6. `model.eval()` and `torch.inference_mode()` calculate output logits.
7. `softmax()` gives normalized model scores. The top three are displayed.

The site gives informational predictions only, not confirmed plant diagnoses or prescribed treatments.

## Based on your files

This package reuses your `Plant Disease Detection.zip` checkpoint, class names, and chart images, and matches the preprocessing in `plant-disease-detection.ipynb`. The extra design image is website decoration only.
