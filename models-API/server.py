from flask import Flask, request, jsonify
import numpy as np
from models import superresolution, detect_layout, lp_ocr, layout_parse_text
from config import exec_net, lp_model
from helper import plot_layout, to_rgb

app = Flask(__name__)


@app.route("/")
def index():
    return '''
    <h1>Welcome to Computer Vision ML Models API!</h1>
    <p><b><i>Developed by Mohamad Oghli<i></b></p>
    '''


@app.route("/models")
def models():
    return '''
     <h3>Currently Available Models</h3>
    <table border='1' style='border-collapse:collapse'>
          <tr>
            <th>Model</th>
            <th>Endpoint</th>
            <th>Parameter</th>
          </tr>
          <tr>
            <td>Super Resolution</td>
            <td>/models/super_resolution</td>
            <td>Image</td>
          </tr>
    </table>
    '''


@app.route('/models/super_resolution', methods=['POST'])
def sr_inference():
    data = request.get_json()
    if 'image' in data:
        req_image = np.array(data['image'], dtype="uint8")
        org_image, sr_image = superresolution(req_image, exec_net)
        return jsonify({"origin": org_image.tolist(), "super": sr_image.tolist()})
    return jsonify({"message": "Sorry, Invalid Image Parameter!"})


@app.route('/models/layout_ocr', methods=['POST'])
def lp_ocr_agent():
    data = request.get_json()
    if 'image' in data:
        req_image = np.array(data['image'], dtype="uint8")
        #req_image = to_rgb(req_image)
        ocr_text = lp_ocr(req_image, 'eng')
        return jsonify({"ocr": ocr_text})
    return jsonify({"message": "Sorry, Invalid Image Parameter!"})


@app.route('/models/layout_parser', methods=['POST'])
def lp_inference():
    data = request.get_json()
    if 'image' in data:
        req_image = np.array(data['image'], dtype="uint8")
        # Convert the image from BGR (cv2 default loading style)
        # to RGB
        req_image = to_rgb(req_image)
        layout = detect_layout(req_image, lp_model)
        lp_image = plot_layout(req_image, layout)
        return jsonify({"layout": lp_image.tolist()})
    return jsonify({"message": "Sorry, Invalid Image Parameter!"})


@app.route('/models/layout_parser/get_text', methods=['POST'])
def lp_get_text():
    data = request.get_json()
    if 'image' in data:
        req_image = np.array(data['image'], dtype="uint8")
        req_image = to_rgb(req_image)
        lp_text = layout_parse_text(req_image, lp_model)
        return jsonify({"layout_text": lp_text})
    return jsonify({"message": "Sorry, Invalid Image Parameter!"})


if __name__ == "__main__":
    app.run()
