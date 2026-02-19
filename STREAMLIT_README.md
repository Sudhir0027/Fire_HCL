# 🔥 Fire Detection Streamlit App - Quick Start

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the model (one-time setup):**
   - Open `notebook8e9d120cda.ipynb` in Jupyter
   - Run all cells to train the model  
   - Cell 16 will save the model as `fire_detection_model.pth`

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

✨ **Real-time Fire Detection**
- Upload images (JPG, PNG)
- Get instant predictions
- View confidence scores
- See probability distribution chart

📊 **Interactive Dashboard**
- Clean, modern UI with Streamlit
- Responsive design
- Device info display (GPU/CPU)
- Model reload button

🎨 **Custom Styling**
- Color-coded predictions (fire/nofire)
- Progress bars for confidence
- Professional layout

## Usage

1. **Upload an image** using the file uploader
2. **View the prediction** with confidence percentage
3. **Check probability distribution** chart below
4. **Reload model** if you retrain it

## Troubleshooting

**Model not loading?**
- Make sure you've trained the model in the notebook first
- Check that `fire_detection_model.pth` exists in the same directory as `app.py`

**GPU not available?**
- The app automatically detects CUDA
- If GPU is not available, it will use CPU (slower but works)

**Import errors?**
- Run `pip install -r requirements.txt` again
- Restart the terminal/command prompt

## Project Structure

```
d:/new/
├── app.py                           # Streamlit app
├── notebook8e9d120cda.ipynb        # Training notebook
├── fire_detection_model.pth        # Trained model weights
├── requirements.txt                # Dependencies
├── submission.csv                  # Batch predictions
└── Fire-Detection/
    ├── 0/                          # No-fire images
    └── 1/                          # Fire images
```

## Model Info

- **Architecture**: ResNet18
- **Classes**: 2 (Fire / No Fire)
- **Input Size**: 224×224 RGB
- **Validation Accuracy**: ~74%

---

**Version**: 1.0 | **Framework**: Streamlit | **Model**: PyTorch
