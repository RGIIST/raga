from flask import Flask, render_template, request, redirect, url_for
import os, numpy
from werkzeug.utils import secure_filename
from prediction import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the uploaded image on the server
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Redirect to the display page with the image path
        return redirect(url_for('display', filename=filename))

    return "Invalid file type. Allowed types: png, jpg, jpeg, gif"

@app.route('/display/<filename>')
def display(filename):
    # Get the image path and pass it to the display template
    image_path = os.path.join('static/results',filename) 
    return render_template('display.html', image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
