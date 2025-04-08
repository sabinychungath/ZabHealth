from flask import Flask, request, jsonify
from flask_cors import CORS
import speech_recognition as sr
import os
import subprocess
from werkzeug.utils import secure_filename
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes with all origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Azure OpenAI
try:
    client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
    client = None

UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def convert_webm_to_wav(input_path, output_path):
    """Convert webm audio file to wav format with improved quality settings."""
    try:
        command = [
            'ffmpeg',
            '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '44100',  # Increased sample rate
            '-af', 'highpass=200,lowpass=3000,volume=2',  # Audio filters for better quality
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        return False

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    try:
        if client is None:
            return jsonify({"error": "Azure OpenAI client not initialized"}), 500

        data = request.json
        if not data or 'text' not in data:
            logger.error("No text provided in request")
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        if not text.strip():
            logger.error("Empty text provided")
            return jsonify({"error": "Empty text provided"}), 400
            
        logger.info("Generating summary for text of length: %d", len(text))

        # Prompt for GPT to generate a medical summary
        prompt = f"""
        Please analyze the following medical conversation and provide a structured summary with the following sections:
        1. Key points (bullet points of the most important information)
        2. Potential diagnosis (if any mentioned)
        3. Recommendations (any suggested actions or treatments)
        4. Follow-up (any mentioned follow-up appointments or future steps)

        Medical conversation:
        {text}
        """

        # Call Azure OpenAI API
        try:
            response = client.chat.completions.create(
                model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                messages=[
                    {"role": "system", "content": "You are a medical professional assistant helping to summarize medical conversations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            # Get the summary text from the response
            summary_text = response.choices[0].message.content
            logger.info("Successfully received Azure OpenAI response")

            # Parse the summary into structured format
            sections = summary_text.split('\n\n')
            summary = {
                'keyPoints': [],
                'diagnosis': None,
                'recommendations': [],
                'followUp': None
            }

            for section in sections:
                if 'Key points:' in section or 'Key Points:' in section:
                    points = section.split('\n')[1:]  # Skip the header
                    summary['keyPoints'] = [point.strip('- ') for point in points if point.strip()]
                elif 'Diagnosis:' in section or 'Potential diagnosis:' in section:
                    diagnosis_text = section.split('\n', 1)[-1].strip('- ')
                    summary['diagnosis'] = diagnosis_text
                elif 'Recommendations:' in section:
                    recommendations = section.split('\n')[1:]  # Skip the header
                    summary['recommendations'] = [rec.strip('- ') for rec in recommendations if rec.strip()]
                elif 'Follow-up:' in section or 'Follow up:' in section:
                    followup_text = section.split('\n', 1)[-1].strip('- ')
                    summary['followUp'] = followup_text

            logger.info("Successfully parsed summary structure")
            return jsonify(summary)

        except Exception as api_error:
            logger.error(f"Azure OpenAI API error: {str(api_error)}")
            return jsonify({"error": f"Error generating summary: {str(api_error)}"}), 500

    except Exception as e:
        logger.error(f"Error in generate_summary: {str(e)}")
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    logger.info("Received transcription request")
    
    if 'file' not in request.files:
        logger.error("No file in request")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("Empty filename")
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save the uploaded webm file
        webm_filename = secure_filename(file.filename)
        webm_path = os.path.join(UPLOAD_FOLDER, webm_filename)
        file.save(webm_path)
        logger.info(f"Saved webm file at {webm_path}")

        # Convert to WAV with improved quality
        wav_path = os.path.join(UPLOAD_FOLDER, 'output.wav')
        if not convert_webm_to_wav(webm_path, wav_path):
            raise Exception("Failed to convert audio format")

        # Initialize recognizer with adjusted settings
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 4000  # Increase energy threshold
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8  # Shorter pause threshold
        
        # Read the audio file and transcribe
        with sr.AudioFile(wav_path) as source:
            logger.info("Reading audio file")
            # Adjust for ambient noise with longer duration
            recognizer.adjust_for_ambient_noise(source, duration=1)
            # Record the audio
            audio = recognizer.record(source)
            logger.info("Performing transcription")
            
            try:
                # Try with default settings first
                transcribed_text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                # If that fails, try with different language settings
                try:
                    transcribed_text = recognizer.recognize_google(audio, language="en-US")
                except sr.UnknownValueError:
                    # If still fails, try with more lenient settings
                    transcribed_text = recognizer.recognize_google(audio, language="en-US", show_all=True)
                    if transcribed_text and 'alternative' in transcribed_text:
                        transcribed_text = transcribed_text['alternative'][0]['transcript']
                    else:
                        raise sr.UnknownValueError("No transcription possible")
            
            logger.info("Transcription successful")
            return jsonify({"transcribed_text": transcribed_text})

    except sr.UnknownValueError:
        logger.error("Speech could not be recognized")
        return jsonify({"error": "Speech could not be recognized. Please try speaking more clearly and closer to the microphone."}), 400
    except sr.RequestError as e:
        logger.error(f"Could not request results: {str(e)}")
        return jsonify({"error": f"Could not request results: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    finally:
        # Clean up temporary files
        for temp_file in [webm_path, wav_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
