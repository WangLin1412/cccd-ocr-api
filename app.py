from flask import Flask, request, jsonify, send_file
from threading import Semaphore
from flask_cors import CORS
import requests, os, uuid, re, unicodedata
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from collections import deque
import time
import threading


# ===== SLOT LIMIT =====
semaphore = threading.Semaphore(2)
# ===== RATE LIMIT =====
REQUEST_LIMIT = 3
TIME_WINDOW = 20  # seconds
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



def auto_rotate_document(image_path, debug=True):
    import cv2
    import numpy as np

    img0 = cv2.imread(image_path)
    if img0 is None:
        return image_path

    def score_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # TEXT MASK
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            25, 15
        )

        h, w = thresh.shape

        # TEXT DENSITY
        top = thresh[:h//2, :]
        bottom = thresh[h//2:, :]

        text_top = np.sum(top) / 255
        text_bottom = np.sum(bottom) / 255

        # EDGE DIRECTION
        edges = cv2.Canny(blur, 50, 150)
        sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)

        vertical_strength = np.sum(np.abs(sobelx))
        horizontal_strength = np.sum(np.abs(sobely))

        # SCORE LOGIC
        score = 0

        # chữ nằm trên là tốt
        if text_top > text_bottom:
            score += (text_top - text_bottom) * 2
        else:
            score -= (text_bottom - text_top) * 2

        # chữ nằm ngang là tốt
        score += (horizontal_strength - vertical_strength)

        return score

    rotations = {
        0: img0,
        90: cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(img0, cv2.ROTATE_180),
        270: cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }

    scores = {}
    for angle, img in rotations.items():
        s = score_image(img)
        scores[angle] = s
        if debug:
            print(f"   ↪ ROTATE CHECK {angle}° → score = {int(s)}")

    # chọn góc tốt nhất
    best_angle = max(scores, key=scores.get)

    # an toàn: chỉ xoay nếu hơn rõ ràng
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        if sorted_scores[0] - sorted_scores[1] < 0.15 * abs(sorted_scores[0]):
            if debug:
                print("⚠️ ROTATE: not confident → keep original")
            return image_path

    if best_angle != 0:
        if debug:
            print(f"✅ ROTATE DONE → chosen angle = {best_angle}°")
        cv2.imwrite(image_path, rotations[best_angle])
    else:
        if debug:
            print("✅ ROTATE: already correct")

    return image_path




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
        print(f"⏱ RATE COUNT = {len(request_times)} / {REQUEST_LIMIT}")

    # ===== SLOT LIMIT 2 USER =====
    acquired = semaphore.acquire(blocking=False)
    if not acquired:
        # ❗ rollback rate limit vì request chưa được xử lý
        with rate_lock:
            if request_times:
                request_times.pop()
    
        return jsonify({
            "error": "Chưa tới lượt bạn!"
        }), 429


    filename = None
    try:
        # ===== VALIDATE FILE =====
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files["image"]
        filename = f"{uuid.uuid4()}.jpg"
        image.save(filename)

        # 🔁 AUTO ROTATE (LOCAL)
        # auto_rotate_document(filename)

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
