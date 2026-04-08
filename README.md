# Vulgar Language Detection via Video

A full end-to-end machine learning pipeline that detects vulgar, toxic, and hateful language directly from video content. The system extracts audio from uploaded video files, transcribes speech to text using OpenAI Whisper, and classifies the text using a Bidirectional LSTM (Bi-LSTM) deep learning model — all through a Flask-based web application.

---

## Features

- **Video-to-Text Pipeline**: Extracts audio from video files and transcribes it using OpenAI Whisper.
- **Deep Learning Classification**: A Bi-LSTM neural network classifies text into five categories.
- **Classical ML Models**: Also trains Logistic Regression, Naive Bayes, Random Forest, and XGBoost as baselines.
- **Segment-Level Analysis**: Breaks down video transcription by time segment, flagging which portions contain problematic language.
- **Web Interface**: A Flask web application for uploading videos and viewing analysis results in real time.
- **CLI Pipeline**: A single `main.py` entry point to run any or all stages of the pipeline.

### Classification Labels

| Label | Description |
|---|---|
| `clean` | Normal, non-offensive speech |
| `profanity` | Swear words and profane language |
| `hate_speech` | Targeted hate based on race, religion, gender, etc. |
| `abuse` | Personal attacks and direct abuse/threats |
| `spam_toxic` | Toxic spam patterns (gaming toxicity, etc.) |

---

## Project Structure

```
vulgar-language-detection/
├── main.py                    # Main CLI entry point for the full pipeline
├── create_sample_video.py     # Helper to create a sample test video (macOS only)
├── requirements.txt           # Python package dependencies
│
├── data/
│   ├── generate_dataset.py    # Generates a synthetic 50,000-sample labelled dataset
│   └── dataset.csv            # Generated dataset (created by generate_dataset.py)
│
├── preprocessing/
│   ├── text_preprocessor.py   # NLTK-based text cleaning and lemmatization
│   └── video_processor.py     # Video audio extraction and Whisper transcription
│
├── models/
│   ├── train.py               # Trains classical ML models (LR, NB, RF, XGBoost)
│   ├── train_dl.py            # Trains the Bi-LSTM deep learning model
│   └── evaluate.py            # Evaluates the best classical model; saves reports and plots
│
├── webapp/
│   ├── app.py                 # Flask web application
│   ├── static/                # Static assets (CSS, JS, uploaded videos)
│   └── templates/
│       ├── index.html         # Main upload and analysis page
│       └── analytics.html     # Analytics dashboard
│
└── outputs/
    ├── models/                # Saved model files (.joblib, .h5)
    ├── figures/               # Evaluation plots (confusion matrix, etc.)
    └── reports/               # Text classification reports
```

---

## Prerequisites

- **Python 3.9+**
- **ffmpeg** (required by MoviePy and Whisper for audio processing)

Install `ffmpeg` via your system package manager:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/manugandham27/vulgar-language-detection-.git
   cd vulgar-language-detection-
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (required for text preprocessing)

   ```python
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. **Download the spaCy language model** (if using spaCy features)

   ```bash
   python -m spacy download en_core_web_sm
   ```

> **Note for Apple Silicon (M1/M2) users**: The `requirements.txt` automatically installs `tensorflow-macos` instead of standard TensorFlow on `darwin/arm64` platforms.

---

## Usage

All pipeline stages are controlled through `main.py`. Use `--help` for a full list of options:

```bash
python main.py --help
```

### Run the Full Pipeline (first-time setup)

This will generate the dataset, train all models, evaluate them, and start the web app:

```bash
python main.py --all
```

### Run Individual Stages

| Command | Description |
|---|---|
| `python main.py --generate` | Generate the synthetic 50k training dataset |
| `python main.py --train` | Train classical ML models (LR, NB, RF, XGBoost) |
| `python main.py --train-dl` | Train the Bi-LSTM deep learning model |
| `python main.py --eval` | Evaluate the best classical model and save reports |
| `python main.py --web` | Start the Flask web application |

### Example Workflow

```bash
# Step 1: Generate dataset
python main.py --generate

# Step 2: Train classical models (optional baseline)
python main.py --train

# Step 3: Train the deep learning model (required for the web app)
python main.py --train-dl

# Step 4: Evaluate classical models
python main.py --eval

# Step 5: Launch the web application
python main.py --web
```

After launching, open your browser and navigate to **http://localhost:5000**.

---

## Web Application

The web application allows you to upload a video file and receive a full toxicity analysis report.

**Supported video formats:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`  
**Maximum upload size:** 500 MB

### How It Works

1. Upload a video through the web interface.
2. The server extracts the audio track from the video using MoviePy.
3. The audio is transcribed into text using OpenAI Whisper (`base` model by default).
4. The full transcription and each individual time segment are classified by the Bi-LSTM model.
5. Results are displayed in real time, highlighting which segments contain vulgar or toxic language.

> The web app requires the deep learning model to be trained first (`python main.py --train-dl`).

---

## Models

### Deep Learning — Bi-LSTM

The primary model used by the web application. Architecture:

```
Embedding(vocab=20000, dim=100, seq_len=100)
  → Bidirectional LSTM(64, return_sequences=True)
  → Dropout(0.3)
  → Bidirectional LSTM(32)
  → Dense(64, ReLU)
  → Dropout(0.3)
  → Dense(5, Softmax)
```

- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Early Stopping**: Patience of 3 epochs on validation loss  
- **Saved to**: `outputs/models/bilstm_model.h5`

### Classical ML Models (Baselines)

All trained inside a `scikit-learn` pipeline with a TF-IDF vectorizer (`max_features=10000`, bigrams):

| Model | Saved As |
|---|---|
| Logistic Regression | `logistic_regression.joblib` |
| Naive Bayes | `naive_bayes.joblib` |
| Random Forest | `random_forest.joblib` |
| XGBoost | `xgboost.joblib` |
| Best model (auto-selected) | `best_model.joblib` |

### Text Preprocessing

The `TextPreprocessor` class (in `preprocessing/text_preprocessor.py`) applies:
- Lowercasing
- URL and mention removal
- Special character stripping
- Stopword removal (keeping negations and pronouns relevant to toxicity detection)
- WordNet lemmatization

---

## Dataset

The dataset is fully synthetic and generated by `data/generate_dataset.py`. It produces **50,000 samples** (10,000 per class) covering the five label categories. The generator uses template-based text generation with vocabulary lists to create diverse examples.

```bash
python main.py --generate
# Output: data/dataset.csv  (50,000 rows: text, label)
```

---

## Creating a Sample Test Video (macOS Only)

The `create_sample_video.py` script uses the macOS `say` command to generate synthetic speech audio and combines it into an `.mp4` file using MoviePy. This is useful for quickly testing the web application pipeline.

```bash
python create_sample_video.py
# Output: sample_test_video.mp4
```

> This script requires macOS because it depends on the `say` text-to-speech command.

---

## Output Artifacts

After running the pipeline, the following files will be created under the `outputs/` directory:

| File | Description |
|---|---|
| `outputs/models/bilstm_model.h5` | Trained Bi-LSTM model |
| `outputs/models/dl_tokenizer.joblib` | Keras tokenizer for DL model |
| `outputs/models/dl_label_encoder.joblib` | Label encoder for DL model |
| `outputs/models/best_model.joblib` | Best classical ML pipeline |
| `outputs/models/label_encoder.joblib` | Label encoder for classical models |
| `outputs/figures/confusion_matrix.png` | Confusion matrix plot |
| `outputs/reports/classification_report.txt` | Full classification report |

---

## Dependencies

Key libraries used in this project:

| Library | Purpose |
|---|---|
| `tensorflow` / `tensorflow-macos` | Bi-LSTM deep learning model |
| `scikit-learn` | Classical ML models, pipelines, evaluation |
| `xgboost` | XGBoost classifier |
| `nltk` | Text preprocessing (lemmatization, stopwords) |
| `openai-whisper` | Speech-to-text transcription |
| `moviepy` | Video audio extraction |
| `Flask` | Web application framework |
| `joblib` | Model serialization |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Evaluation visualizations |
| `imbalanced-learn` | Handling class imbalance |
| `tqdm` | Progress bars |

Install all dependencies with:

```bash
pip install -r requirements.txt
```
