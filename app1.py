from flask import Flask, render_template_string, request
import uuid, os
from moviepy.editor import VideoFileClip
import yt_dlp
from speechbrain.pretrained.interfaces import foreign_class
from werkzeug.utils import secure_filename
from pydub import AudioSegment

app = Flask(__name__)
UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load SpeechBrain model
classifier = foreign_class(
    source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)

# Mapping model labels to human-readable names
ACCENT_LABELS = {
    "us": "American",
    "gb": "British",
    "au": "Australian",
    "ng": "Nigerian",
    "gh": "Ghanaian",
    "za": "South African",
    "ke": "Kenyan",
    "ph": "Philippine",
    "in": "Indian",
    "ie": "Irish",
    "ca": "Canadian",
    "nz": "New Zealand",
    "jm": "Jamaican",
    "tt": "Trinidadian",
    "sg": "Singaporean",
    "my": "Malaysian"
}

# HTML template
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
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
      <input type="text" name="url" id="url" placeholder="Enter public video URL (optional)">
      <input type="file" name="file" id="file" accept=".mp4,.wav">
      <button type="submit" id="submitBtn">Analyze</button>
    </form>

    <div id="loader" class="loader">
      ⏳ Processing... please wait.
    </div>

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
      const url = document.getElementById("url").value;
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
      if (!validateInput()) return false;
      document.getElementById("submitBtn").disabled = true;
      document.getElementById("loader").classList.add("show");
    });
  </script>
</body>
</html>
"""

def download_video(url):
    video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.mp4")
    ydl_opts = {
        'outtmpl': video_path,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return video_path

def extract_audio(video_file):
    audio_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.wav")
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(audio_path)
    clip.close()
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(audio_path, format="wav")
    return audio_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        file = request.files.get("file")
        try:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                audio_path = file_path if file_path.endswith('.wav') else extract_audio(file_path)
                if file_path.endswith('.mp4'): os.remove(file_path)
            elif url:
                video_path = download_video(url)
                audio_path = extract_audio(video_path)
                os.remove(video_path)
            else:
                raise Exception("Please provide a valid URL or file.")

            out_prob, score, index, label = classifier.classify_file(audio_path)
            label_str = label[0] if isinstance(label, list) else label
            readable_label = ACCENT_LABELS.get(label_str.lower(), label_str.upper())
            os.remove(audio_path)

            result = {
                "accent": readable_label,
                "confidence": round(float(score) * 100, 2),
                "explanation": f"Detected using SpeechBrain model trained on 16 English accents. Code: '{label_str}'"
            }
            return render_template_string(HTML_TEMPLATE, result=result)
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, result={
                "accent": "Error",
                "confidence": 0,
                "explanation": str(e)
            })

    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True)


