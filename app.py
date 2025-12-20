from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests, os, uuid, re, unicodedata
import pandas as pd

app = Flask(__name__)
CORS(app)

OCR_API_KEY = os.environ.get("OCR_API_KEY")

# ===============================
# HÀM HẬU XỬ LÝ CCCD (DÙNG CHUNG)
# ===============================
def clean_cccd_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    # 1. Normalize Unicode
    text = unicodedata.normalize("NFKC", raw_text)

    # 2. Fix lỗi OCR phổ biến (áp dụng cho MỌI CCCD)
    replaces = {
        "CONG HOA": "CỘNG HÒA",
        "Hél": "HỘI",
        "CHÜ": "CHỦ",
        "NGHiA": "NGHĨA",
        "VlÉ:r": "VIỆT",

        "Döc lap": "Độc lập",
        "do -": "-",
        "Henh phüc": "Hạnh phúc",

        "GÄN CU'dc CONG DAN": "CĂN CƯỚC CÔNG DÂN",
        "GAN CUOC CONG DAN": "CĂN CƯỚC CÔNG DÂN",

        "s6:": "Số:",
        "HQ tén": "Họ và tên",

        "Ngåy, thång, näm sinh": "Ngày sinh",
        "Ciöi tinh": "Giới tính",
        "Qu6ctich": "Quốc tịch",
        "Qué quån": "Quê quán",
        "Ndi thddng trü": "Nơi thường trú",

        "Viet Nam": "Việt Nam"
    }

    for k, v in replaces.items():
        text = text.replace(k, v)

    # 3. Xóa ký tự rác
    text = re.sub(r"[`~^*_]", "", text)
    text = re.sub(r"\s{2,}", " ", text)

    # 4. Chuẩn hóa dòng
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    output = []
    for line in lines:
        # bỏ dòng ngày cấp không cần thiết
        if re.match(r"^\d{2}/\d{2}/\d{4}$", line):
            continue

        # chuẩn số CCCD
        if "Số:" in line:
            m = re.search(r"\d{12}", line)
            if m:
                output.append(f"Số: {m.group()}")
            continue

        output.append(line)

    return "\n".join(output)


# ===============================
# ROUTES
# ===============================
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
                "language": "vie",
                "OCREngine": "2"
            },
            timeout=60
        )

        result = response.json()

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

        raw_text = result["ParsedResults"][0].get("ParsedText", "")
        text = clean_cccd_text(raw_text)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filename):
            os.remove(filename)

    # Xuất Excel
    excel_name = f"{uuid.uuid4()}.xlsx"
    df = pd.DataFrame([{"CCCD_TEXT": text}])
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
