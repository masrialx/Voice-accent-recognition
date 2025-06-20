English Accent Analyzer Web Application
A web-based application that analyzes English accents from video or audio inputs using state-of-the-art speech processing and machine learning models. Users can provide either a public video URL or upload a video/audio file, and the app extracts speech to classify the speaker’s accent with confidence scores.

Table of Contents
Project Overview

Features

Technologies Used

Installation

Usage

Project Structure

Model Details

Limitations & Future Work

License

Contact

Project Overview
This Flask-based application provides an intuitive interface to analyze and classify English accents from video or audio data. It supports both URL-based downloads from a wide variety of platforms (YouTube, Loom, Google Drive, and others supported by yt-dlp) and direct file uploads (MP4 and WAV formats).

The backend extracts audio from video files, processes the audio with a pre-trained Wav2Vec2 model fine-tuned for accent classification, and returns the predicted accent with confidence levels and explanations.

Features
Multi-source video download: Seamlessly downloads videos from various public URLs via yt-dlp.

Direct file upload: Upload MP4 or WAV files if the URL download fails or is unavailable.

Audio extraction: Extracts high-quality 16kHz mono WAV audio from video files.

Accent classification: Uses a Wav2Vec2 Transformer-based model to classify accents into categories such as American, British, Indian, Australian, and Canadian.

User-friendly UI: Simple web interface with clear forms, validation, and real-time processing notifications.

Error handling & user guidance: Informs users if video download from URL fails and recommends direct file upload.

Clean temporary storage: Automatically manages temporary video/audio files.

Technologies Used
Python 3.8+ — Core programming language

Flask — Lightweight web framework

yt-dlp — Robust media downloader supporting many platforms

moviepy — Video processing and audio extraction

pydub — Audio processing

torch & torchaudio — Deep learning and audio I/O

transformers (Hugging Face) — Pre-trained Wav2Vec2 model for accent classification

HTML/CSS/JavaScript — Frontend form, validation, and user interaction

Installation
Prerequisites
Python 3.8 or higher installed

ffmpeg installed and available in your system PATH (required by moviepy and yt-dlp)

Setup steps
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/accent-analyzer.git
cd accent-analyzer
Create and activate a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
python app.py
Access the app:

Open your browser and go to http://127.0.0.1:5000

Usage
Enter a public video URL in the input box (e.g., a YouTube, Loom, or Google Drive video link) or upload a local video/audio file (MP4 or WAV).

Click Analyze.

If the video is downloaded and processed successfully, the app will display the predicted English accent and the confidence score.

If the URL download fails, a notification will recommend uploading the video file directly.

Uploaded video files are automatically processed to extract audio for analysis.

Project Structure
bash
Copy
Edit
├── app.py                # Main Flask application script
├── requirements.txt      # Python dependencies
├── temp/                 # Temporary storage for uploaded and downloaded media
└── README.md             # Project documentation (this file)
Model Details
This project uses the dima806/english_accents_classification model from Hugging Face. It is a fine-tuned Wav2Vec2 Transformer model designed to classify English speech accents into several categories:

American (US)

British (England)

Indian

Australian

Canadian

The model takes 16kHz mono WAV audio as input and outputs probabilities for each accent class. The highest scoring class is presented to the user along with a confidence percentage.

Limitations & Future Work
Supported formats: Only MP4 (video) and WAV (audio) uploads supported currently.

URL downloads: Dependent on yt-dlp’s support and availability; private or restricted content may fail.

Performance: Large video files may cause delays due to download and processing time.

UI improvements: Can be enhanced for better user experience and responsiveness.

Additional accents: Future expansion to include more accent classes and languages.

API support: Can be extended to provide REST API endpoints for programmatic usage.

