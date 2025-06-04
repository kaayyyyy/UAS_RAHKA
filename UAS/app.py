from flask import Flask, render_template, request, redirect, url_for, Response
from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = Image.open(filepath)
    img_processed, plot_filename = improve_image_quality(img)

    processed_filename = 'processed_' + file.filename
    processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    img_processed.save(processed_filepath)

    return render_template('index.html',
                           original_filename=file.filename,
                           processed_filename=processed_filename,
                           plot_filename=plot_filename)

def improve_image_quality(img):
    # Konversi ke format BGR OpenCV dari PIL Image RGB
    img_bgr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)

    # Proses HSV masking untuk warna biru
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue1 = np.array([110, 50, 50])
    blue2 = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, blue1, blue2)

    # Morphological opening untuk hilangkan noise
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # --- Bagian tambahan: baca citra grayscale dari file 'struktur.png' ---
    # Jika file 'struktur.png' ada di folder static/uploads, baca dan proses
    struktur_path = os.path.join(app.config['UPLOAD_FOLDER'], 'struktur.png')
    if os.path.exists(struktur_path):
        img_struktur = cv2.imread(struktur_path, 0)
    else:
        # Kalau tidak ada, fallback ke mask hasil HSV sebagai input erosi/dilasi
        img_struktur = opening

    # Operasi erosi dan dilasi
    img_erosion = cv2.erode(img_struktur, kernel, iterations=1)
    img_dilation = cv2.dilate(img_struktur, kernel, iterations=1)

    # Buat plot lengkap hasil proses
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    ax = axes.ravel()

    ax[0].imshow(img_struktur, cmap='gray')
    ax[0].set_title("Citra Input")
    ax[0].axis('off')
    ax[1].hist(img_struktur.ravel(), bins=256)
    ax[1].set_title("Histogram Citra Input")

    ax[2].imshow(img_erosion, cmap='gray')
    ax[2].set_title("Citra Output Erosi")
    ax[2].axis('off')
    ax[3].hist(img_erosion.ravel(), bins=256)
    ax[3].set_title("Histogram Citra Output Erosi")

    ax[4].imshow(img_dilation, cmap='gray')
    ax[4].set_title("Citra Output Dilasi")
    ax[4].axis('off')
    ax[5].hist(img_dilation.ravel(), bins=256)
    ax[5].set_title("Histogram Citra Output Dilasi")

    plt.tight_layout()
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'plot_full.png')
    plt.savefig(plot_path)
    plt.close(fig)

    # Buat gambar output utama (hasil masking dan opening)
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=opening)
    final_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    return final_image, 'plot_full.png'


# Webcam streaming tetap seperti sebelumnya
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blue1 = np.array([110, 50, 50])
            blue2 = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, blue1, blue2)
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            result = cv2.bitwise_and(frame, frame, mask=opening)

            ret, buffer = cv2.imencode('.jpg', result)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/webcam')
def webcam():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
