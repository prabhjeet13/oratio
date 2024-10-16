from flask import Flask,render_template, request,jsonify,send_file
import os
from werkzeug.utils import secure_filename
from STSModel import speech_to_text, translate_text, text_to_speech
from flask_cors import CORS
app = Flask(__name__)
app.secret_key = '131313' 
CORS(app)
@app.route('/')
def homepage():
    return render_template('index.html')


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/oratio',methods=['POST'])
def translate_speech_to_speech():
    # three models one by one STT TT TTS
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file part in the request"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            transcription = speech_to_text(file_path)
            os.remove(file_path)
            # print(transcription)
            # print('fgefgegfywgyfgwe',transcription)
            target_lang = request.form.get('target_lang')
            if transcription:
                  output_file = 'output_audio.wav'
                  text_to_speech(transcription,target_lang,output_file)
                  return send_file(output_file, mimetype='audio/wav')
                    # os.remove(output_file)  # Delete after sending
    except Exception as e:
        return jsonify({"error" : "an unexpected error"}),500


if __name__ == "__main__":
    # app.run(port = 5000,debug=True)  
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))