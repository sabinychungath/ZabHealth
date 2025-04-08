from flask import Flask, request, jsonify
import speech_recognition as sr

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)
    
    try:
        transcribed_text = recognizer.recognize_google(audio)
        return jsonify({"transcribed_text": transcribed_text})
    except sr.UnknownValueError:
        return jsonify({"error": "Speech could not be recognized"}), 400
    except sr.RequestError:
        return jsonify({"error": "Speech recognition service unavailable"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
