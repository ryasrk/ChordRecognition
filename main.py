import os
import json
import tempfile
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import numpy as np
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor
from madmom.processors import SequentialProcessor

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXT = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'mp4', 'mov', 'avi', 'mkv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def extract_audio(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(44100).set_channels(1)
    audio.export(output_path, format='wav')

def recognize_chords(wav_path):
    dcp    = DeepChromaProcessor()
    decode = DeepChromaChordRecognitionProcessor()
    proc   = SequentialProcessor([dcp, decode])
    chords = proc(wav_path)
    # Convert to list of dicts
    timeline = [
        {'start': float(s), 'end': float(e), 'label': l}
        for s,e,l in chords
    ]
    return timeline

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('media')
        if not file or not allowed_file(file.filename):
            return "Invalid file", 400

        filename = secure_filename(file.filename)
        media_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(media_path)

        # Extract to WAV and run chord recognition
        wav_file = media_path + '.wav'
        extract_audio(media_path, wav_file)
        timeline = recognize_chords(wav_file)

        # Save chord data as JSON alongside media
        json_path = media_path + '.json'
        with open(json_path, 'w') as f:
            json.dump(timeline, f)

        return render_template('player.html',
                               media_url=f'/media/{filename}',
                               chord_url=f'/media/{filename}.json',
                               ext=filename.rsplit('.',1)[1].lower())
    return render_template('index.html')


@app.route('/media/<path:filename>')
def media(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/chords', methods=['GET'])
def api_chords():
    """ Alternate JSON endpoint if you want to fetch by query param. """
    media_fn = request.args.get('media')
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(media_fn) + '.json')
    if not os.path.exists(json_path):
        return jsonify([]), 404
    with open(json_path) as f:
        timeline = json.load(f)
    return jsonify(timeline)


if __name__ == '__main__':
    app.run(debug=True)
