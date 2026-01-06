from flask import Flask, request, jsonify, send_file
from threading import Semaphore
from flask_cors import CORS
import requests, os, uuid, re, unicodedata
import pandas as pd
from PIL import Image
import cv2
import pytesseract
import numpy as np
from collections import deque
import time
import threading


# ===== SLOT LIMIT =====
semaphore = threading.Semaphore(2)
# ===== DEBUG CONCURRENCY =====
active_requests = 0
active_lock = threading.Lock()
# ===== RATE LIMIT =====
REQUEST_LIMIT = 10
TIME_WINDOW = 60  # seconds
request_times = deque()
rate_lock = threading.Lock()

app = Flask(__name__)

# ✅ CORS CHUẨN CHO WORDPRESS + FETCH
CORS(
    app,
    resources={r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }}
)

OCR_API_KEY = os.environ.get("OCR_API_KEY")

# ===============================
# HÀM HẬU XỬ LÝ CCCD (DÙNG CHUNG)
# ===============================
def clean_cccd_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    # 1. Chuẩn hóa Unicode
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

    # 3. Loại ký tự rác
    text = re.sub(r"[`~^*_]", "", text)
    text = re.sub(r"\s{2,}", " ", text)

    # 4. Chuẩn hóa dòng
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    output = []

    for line in lines:
        # Bỏ dòng ngày cấp (nếu có)
        if re.match(r"^\d{2}/\d{2}/\d{4}$", line):
            continue

        # Chuẩn số CCCD
        if "Số:" in line:
            m = re.search(r"\d{12}", line)
            if m:
                output.append(f"Số: {m.group()}")
            continue

        output.append(line)

    return "\n".join(output)

def auto_rotate_image(image_path):
    """
    Tự động xoay ảnh CCCD về đúng chiều
    """
    image = cv2.imread(image_path)
    if image is None:
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        osd = pytesseract.image_to_osd(gray)
        rotate_angle = 0

        if "Rotate: 90" in osd:
            rotate_angle = 90
        elif "Rotate: 180" in osd:
            rotate_angle = 180
        elif "Rotate: 270" in osd:
            rotate_angle = 270

        if rotate_angle != 0:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -rotate_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(image_path, rotated)

    except Exception as e:
        print("Auto-rotate failed:", e)

# ===============================
# ROUTES
# ===============================
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        return response
@app.route("/", methods=["GET"])
def home():
    return "CCCD OCR API is running"

# ✅ BẮT BUỘC CÓ OPTIONS
@app.route("/ocr", methods=["POST", "OPTIONS"])
def ocr():

    # ===== CORS PREFLIGHT =====
    if request.method == "OPTIONS":
        return "", 200

    # ===== RATE LIMIT 10 REQ / PHÚT =====
    now = time.time()
    with rate_lock:
        while request_times and now - request_times[0] > TIME_WINDOW:
            request_times.popleft()

        if len(request_times) >= REQUEST_LIMIT:
            return jsonify({
                "error": "Hệ thống đang bận, vui lòng thử lại sau"
            }), 429

        request_times.append(now)

    # ===== SLOT LIMIT 2 USER =====
    acquired = semaphore.acquire(blocking=False)
    if not acquired:
        return jsonify({
            "error": "Chưa tới lượt bạn!"
        }), 429
        
    with active_lock:
        global active_requests
        active_requests += 1
        print("🟢 ACTIVE REQUESTS =", active_requests)

    # ⛔ CHỈ ĐỂ TEST – GIỮ REQUEST LẠI 10s
    time.sleep(10)

    filename = None
    try:
        # ===== VALIDATE FILE =====
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files["image"]
        filename = f"{uuid.uuid4()}.jpg"
        image.save(filename)

        # ✅ AUTO ROTATE CCCD
        auto_rotate_image(filename)

        # ===== OCR.SPACE (HARD TIMEOUT) =====
        try:
            response = requests.post(
                "https://api.ocr.space/parse/image",
                files={"file": open(filename, "rb")},
                data={
                    "apikey": OCR_API_KEY,
                    "language": "auto",
                    "OCREngine": "2"
                },
                timeout=(5, 60)  # 🔥 connect 5s, read 60s
            )
        except requests.exceptions.ConnectTimeout:
            return jsonify({
                "error": "Không kết nối được OCR, vui lòng thử lại"
            }), 504
        except requests.exceptions.ReadTimeout:
            return jsonify({
                "error": "OCR xử lý quá lâu, vui lòng gửi lại ảnh"
            }), 504

        # 🚨 OCR.SPACE QUÁ TẢI
        if response.status_code == 429:
            return jsonify({
                "error": "OCR đang quá tải, vui lòng thử lại sau"
            }), 429

        result = response.json()

        # 🚨 OCR SPACE BÁO LỖI
        if result.get("IsErroredOnProcessing"):
            return jsonify({
                "error": "OCR failed",
                "message": result.get("ErrorMessage", "Unknown error")
            }), 400

        # 🚨 OCR KHÔNG ĐỌC ĐƯỢC CHỮ
        parsed = result.get("ParsedResults")
        if not parsed or not parsed[0].get("ParsedText"):
            return jsonify({
                "error": "Không nhận diện được chữ trong ảnh. Vui lòng chụp rõ hơn."
            }), 400

        raw_text = parsed[0]["ParsedText"]
        text = clean_cccd_text(raw_text)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        with active_lock:
            active_requests -= 1
            print("🔵 REQUEST FINISHED → ACTIVE =", active_requests)
        # 🔓 NHẢ SLOT + DỌN FILE
        if acquired:
            semaphore.release()
        if filename and os.path.exists(filename):
            os.remove(filename)

    # ===== EXPORT EXCEL =====
    excel_name = f"{uuid.uuid4()}.xlsx"
    df = pd.DataFrame([{"CCCD_TEXT": text}])
    df.to_excel(excel_name, index=False)

    return jsonify({
        "text": text,
        "excel_url": f"/download/{excel_name}"
    })

@app.route("/download/<name>", methods=["GET"])
def download(name):
    if os.path.exists(name):
        return send_file(name, as_attachment=True)
    return "Not found", 404
