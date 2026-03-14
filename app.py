from flask import Flask, Response, request, jsonify, send_file
from flask_cors import cross_origin, CORS
import os
import uuid

from model import fusionVGG19, dilationInceptionModule
from imagem_service import ImagemService, desenhar_pontos

service = ImagemService("models/Best_Model400it.pt")

app = Flask(__name__)
CORS(app)

WORK_DIR = os.path.abspath(os.path.dirname(__file__))


@app.route("/process-image", methods=["POST"])
@cross_origin()
def processar() -> Response:
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_temp_path = os.path.join(WORK_DIR, f"temp_{uuid.uuid4().hex}.png")
    img_overlay_path = os.path.join(WORK_DIR, f"overlay_{uuid.uuid4().hex}.png")
    file.save(img_temp_path)

    try:
        coords_list, angles = service.predict(img_temp_path)
        desenhar_pontos(img_temp_path, coords_list, img_overlay_path)
        return jsonify(
            {
                "coords": coords_list,
                "angles": angles,
                "image_with_overlay_path": img_overlay_path,
            }
        )
    except Exception as e:
        if os.path.exists(img_overlay_path):
            os.remove(img_overlay_path)
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(img_temp_path):
            os.remove(img_temp_path)


@app.route("/download-image/<filename>")
def download_imagem(filename):
    safe_name = os.path.basename(filename)
    safe_path = os.path.join(WORK_DIR, safe_name)
    if not os.path.exists(safe_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(safe_path, mimetype="image/png", as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
