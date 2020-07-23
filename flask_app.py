from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from ai import get_yolo_net, yolo_forward, yolo_save_img
import cv2
import numpy as np

# where we will store images
UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# load the NN to memory
here = os.getcwd()
names_path = os.path.join(here, 'yolo', 'currency_obj.names')
with open(names_path, "r") as f:
    LABELS = f.read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weights_path = os.path.join(here, 'yolo', 'currency_yolov3_final.weights')
cfg_path = os.path.join(here, 'yolo', 'currency_yolov3.cfg')
net = get_yolo_net(cfg_path, weights_path)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# routes definitions
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(render_template('home.html', error="Error: No file part"))

        file = request.files['file']
        # if user does not select file, browser also
        # submits an empty part without filename
        if file.filename == '':
            return redirect(render_template('home.html', error="Error: No selected file"))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))

        return redirect(render_template('home.html', error="Error: Wrong file type"))

    return render_template('home.html', error="‎")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # read image file and make prediction
    here = os.getcwd()
    image_path = os.path.join(here, app.config['UPLOAD_FOLDER'], filename)

    try:
        image = cv2.imread(image_path)
        (class_ids, labels, boxes, confidences) = yolo_forward(net, LABELS, image, confidence_level=0.3)
    except Exception:
        return redirect(url_for('home'))

    # format data for template rendering
    # found emotions, save images with bounding boxes.
    if len(class_ids) > 0:
        new_filename = 'yolo_' + filename
        file_path = os.path.join(here, app.config['UPLOAD_FOLDER'], new_filename)
        yolo_save_img(image, class_ids, boxes, labels, confidences, COLORS, file_path=file_path)

        # help function to format result sentences.
        def and_syntax(alist):
            if len(alist) == 1:
                alist = "".join(alist)
                return alist
            elif len(alist) == 2:
                alist = " and ".join(alist)
                return alist
            elif len(alist) > 2:
                alist[-1] = "and " + alist[-1]
                alist = ", ".join(alist)
                return alist
            else:
                return

        # confidences: rounding and changing to percent, putting in function
        format_confidences = []
        for percent in confidences:
            format_confidences.append(str(round(percent*100)) + '%')
        format_confidences = and_syntax(format_confidences)
        # labels: sorting and capitalizing, putting into function
        labels = set(labels)  # Out of order??
        labels = [label.capitalize() for label in labels]
        labels = and_syntax(labels)

        # return template with data
        return render_template('results.html',
                               confidences=format_confidences,
                               labels=labels,
                               old_filename=filename,
                               filename=new_filename)
    else:
        return render_template('results.html',
                               labels='No Detection',
                               old_filename=filename,
                               filename=filename)


@app.route('/files/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('home.html', error='Error: File too large'), 413


@app.errorhandler(404)
def error_not_found(error):
    return redirect(url_for('home'))
