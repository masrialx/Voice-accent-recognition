from flask import Flask, render_template_string, request
import uuid
import os
from moviepy.editor import VideoFileClip
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import yt_dlp

app = Flask(__name__)
UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_ID = "dima806/english_accents_classification"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)

LABELS = ["us", "england", "indian", "australia", "canada"]
PRETTY = {
    "us": "American",
    "england": "British",
    "indian": "Indian",
    "australia": "Australian",
    "canada": "Canadian"
}

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Accent Analyzer</title>
  <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f3f3f3;
        padding: 2rem;
        color: #333;
    }
    .container {
        max-width: 600px;
        margin: auto;
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        position: relative;
    }
    input, button {
        width: 100%;
        padding: 0.75rem;
        margin: 1rem 0;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    button {
        background-color: #007bff;
        color: white;
        border: none;
        cursor: pointer;
    }
    button:disabled {
        background-color: #999;
        cursor: not-allowed;
    }
    button:hover:enabled {
        background-color: #0056b3;
    }
    .result {
        background-color: #e7f3ff;
        padding: 1rem;
        border-left: 4px solid #007bff;
        margin-top: 1rem;
    }
    .notification {
        background-color: #fff3cd;
        border-left: 4px solid #ffecb5;
        padding: 1rem;
        margin-top: 1rem;
        color: #856404;
    }
    .loader {
        display: none;
        text-align: center;
        margin-top: 10px;
    }
    .loader.show {
        display: block;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🎤 English Accent Analyzer</h1>
    <form method="POST" action="/" enctype="multipart/form-data" id="analyzeForm">
      <input type="text" name="url" id="url" placeholder="Enter YouTube URL (optional)" />
      <input type="file" name="file" id="file" accept=".mp4,.wav" />
      <button type="submit" id="submitBtn">Analyze</button>
    </form>

    <div id="loader" class="loader">⏳ Processing... please wait.</div>

    {% if notification %}
    <div class="notification">
      ⚠️ {{ notification }}
    </div>
    {% endif %}

    {% if result %}
    <div class="result">
      <h2>Result:</h2>
      <p><strong>Accent:</strong> {{ result.accent }}</p>
      <p><strong>Confidence:</strong> {{ result.confidence }}%</p>
      <p><strong>Explanation:</strong> {{ result.explanation }}</p>
    </div>
    {% endif %}
  </div>

  <script>
    function validateInput() {
      const url = document.getElementById("url").value.trim();
      const file = document.getElementById("file").files[0];
      if (!url && !file) {
        alert("Please provide a URL or upload a file!");
        return false;
      }
      if (url) {
        const pattern = /^(https?:\/\/)?([\w\-\.]+)\.([a-z]{2,6})(\/.*)?$/;
        if (!pattern.test(url)) {
          alert("Please enter a valid URL!");
          return false;
        }
      }
      if (file && !file.name.match(/\.(mp4|wav)$/i)) {
        alert("Please upload an MP4 or WAV file!");
        return false;
      }
      return true;
    }

    document.getElementById("analyzeForm").addEventListener("submit", function (e) {
      if (!validateInput()) {
        e.preventDefault();
        return false;
      }
      document.getElementById("submitBtn").disabled = true;
      document.getElementById("loader").classList.add("show");
    });
  </script>
</body>
</html>
"""

def download_video(url):
    filename = f"{uuid.uuid4()}.mp4"
    video_path = os.path.join(UPLOAD_FOLDER, filename)

    ydl_opts = {
        'format': 'best',
        'outtmpl': video_path,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(video_path):
        raise Exception("Failed to download video or file not found")

    return video_path

def extract_audio(video_file):
    audio_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.wav")
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    clip.close()

    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(audio_path, format="wav")
    return audio_path

def classify_accent(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    inputs = feature_extractor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted = torch.argmax(logits, dim=-1).item()
    confidence = torch.softmax(logits, dim=-1)[0][predicted].item()
    label = LABELS[predicted]
    return PRETTY.get(label, label), round(confidence * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    notification = None
    if request.method == "POST":
        url = request.form.get("url")
        file = request.files.get("file")
        try:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                audio_path = file_path if file_path.endswith(".wav") else extract_audio(file_path)
                if file_path.endswith(".mp4"):
                    os.remove(file_path)
            elif url:
                video_path = download_video(url)
                audio_path = extract_audio(video_path)
                os.remove(video_path)
            else:
                raise Exception("Please provide a valid URL or upload a file.")

            accent, confidence = classify_accent(audio_path)
            os.remove(audio_path)

            result = {
                "accent": accent,
                "confidence": confidence,
                "explanation": "Classification done using Wav2Vec2 English Accent Model."
            }
            return render_template_string(HTML_TEMPLATE, result=result, notification=notification)
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, result={
                "accent": "Error",
                "confidence": 0,
                "explanation": str(e)
            }, notification=notification)

    return render_template_string(HTML_TEMPLATE, notification=None)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
