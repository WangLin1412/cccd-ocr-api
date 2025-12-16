from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests, os, uuid
import pandas as pd

app = Flask(__name__)
CORS(app)

OCR_API_KEY = os.environ.get("OCR_API_KEY")

@app.route("/")
def home():
    return "CCCD OCR API is running"

@app.route("/ocr", methods=["POST"])
def ocr():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    filename = f"{uuid.uuid4()}.jpg"
    image.save(filename)

    try:
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": open(filename, "rb")},
            data={
                "apikey": OCR_API_KEY,
                "language": "vnm"
            },
            timeout=60
        )

        result = response.json()

        # 👉 CHECK LỖI OCR.SPACE
        if result.get("IsErroredOnProcessing"):
            return jsonify({
                "error": "OCR failed",
                "message": result.get("ErrorMessage", "Unknown error"),
                "details": result
            }), 400

        if "ParsedResults" not in result:
            return jsonify({
                "error": "Invalid OCR response",
                "details": result
            }), 400

        text = result["ParsedResults"][0].get("ParsedText", "")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filename):
            os.remove(filename)

    excel_name = f"{uuid.uuid4()}.xlsx"
    df = pd.DataFrame([{"content": text}])
    df.to_excel(excel_name, index=False)

    return jsonify({
        "text": text,
        "excel_url": f"/download/{excel_name}"
    })


@app.route("/download/<name>")
def download(name):
    if os.path.exists(name):
        return send_file(name, as_attachment=True)
    return "Not found", 404
