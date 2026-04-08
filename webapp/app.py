import os
import sys
import uuid
import threading
import joblib
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.video_processor import VideoProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Resources
vp = None
ml_model = None
label_encoder = None
tokenizer = None

# Async Task Tracking
tasks = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_resources():
    global vp, ml_model, label_encoder, tokenizer
    print("Loading resources... this may take a minute.")
    if vp is None:
        vp = VideoProcessor("base")
        
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'models')
    model_path = os.path.join(models_dir, 'bilstm_model.h5')
    encoder_path = os.path.join(models_dir, 'dl_label_encoder.joblib')
    tokenizer_path = os.path.join(models_dir, 'dl_tokenizer.joblib')
    
    if os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(tokenizer_path):
        if ml_model is None:
            ml_model = tf.keras.models.load_model(model_path)
            label_encoder = joblib.load(encoder_path)
            tokenizer = joblib.load(tokenizer_path)
            print("Successfully loaded Deep Learning model!")
    else:
        print("Warning: Deep Learning Models not found. Please train with --train-dl first.")

@app.before_request
def setup():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    load_resources()

def process_video_task(task_id, video_path, filename):
    try:
        tasks[task_id]['status'] = 'Extracting Audio & Transcribing'
        tasks[task_id]['progress'] = 25
        
        # 1. Transcribe audio
        transcription = vp.process_video(video_path)
        if not transcription or not transcription["text"]:
            raise Exception("No speech detected or extraction failed.")
            
        tasks[task_id]['status'] = 'Analyzing Text via Deep Learning'
        tasks[task_id]['progress'] = 75
            
        def predict_text(text):
            seq = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
            pred_probs = ml_model.predict(padded, verbose=0)
            class_idx = tf.argmax(pred_probs, axis=1).numpy()[0]
            confidence = float(pred_probs[0][class_idx])
            return label_encoder.inverse_transform([class_idx])[0], confidence

        # 2. Analyze entire text
        full_text = transcription["text"]
        full_label, full_conf = predict_text(full_text)
        
        # 3. Analyze segments
        segments_data = []
        for seg in transcription["segments"]:
            seg_text = seg["text"].strip()
            if seg_text:
                label, conf = predict_text(seg_text)
                segments_data.append({
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg_text,
                    "classification": label,
                    "confidence": round(conf * 100, 1),
                    "is_vulgar": label != "clean"
                })
                
        tasks[task_id]['status'] = 'Complete'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['result'] = {
            "video_url": f"/static/uploads/{filename}",
            "overall_classification": full_label,
            "overall_confidence": round(full_conf * 100, 1),
            "full_text": full_text,
            "segments": segments_data
        }
    except Exception as e:
        tasks[task_id]['status'] = 'Error'
        tasks[task_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if ml_model is None:
        return jsonify({"error": "Deep Learning Model is not loaded. Train with --train-dl first."}), 500
        
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            'status': 'Uploaded, preparing processing...',
            'progress': 10,
            'result': None
        }
        
        thread = threading.Thread(target=process_video_task, args=(task_id, video_path, filename))
        thread.start()
        
        return jsonify({"task_id": task_id})
        
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/status/<task_id>', methods=['GET'])
def task_status(task_id):
    if task_id in tasks:
        return jsonify(tasks[task_id])
    return jsonify({"error": "Task not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
