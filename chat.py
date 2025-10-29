import os
import json
import requests
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

# =======================
# CONFIG
# =======================
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://192.168.2.37:11434")
GEN_URL     = f"{OLLAMA_BASE}/api/generate"
TAGS_URL    = f"{OLLAMA_BASE}/api/tags"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma3:4b")

SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยสาวให้ข้อมูลเกี่ยวกับวิทยาลัยการอาชีพปราจีนบุรี "
    "คุณเป็น AI ของวิทยาลัยการอาชีพปราจีนบุรี คุณเป็นผู้หญิง และจะตอบกลับแบบผู้หญิงเท่านั้น "
    "คุณจะไม่พูดคำว่าคับ หรือ ครับ "
    "คุณตอบคำถามแบบสั้น ๆ กระชับ และให้ข้อมูลครบถ้วน "
    "คุณจะไม่อ่านหรือสนใจเครื่องหมาย * "
	"คุณจะไม่เพิ่มเติมข้อมูลเอง คุณจะต้องพูดตามข้อมูลที่เรามีให้อย่างชัดเจน"
    "สามารถให้ข้อมูลเกี่ยวกับประวัติวิทยาลัยได้\n\n"

    "ข้อมูลวิทยาลัยการอาชีพปราจีนบุรี:\n"
    "ก่อตั้งเมื่อวันที่ 16 มิถุนายน 2536\n"
    "ที่ตั้ง: 306/1 ถ.ราษฎรดำริ ต.หน้าเมือง อ.เมือง จ.ปราจีนบุรี 25000\n"
    "โทร: 037-212-220\n"
    "อีเมล: piccollege21359@gmail.com\n"
    "เว็บไซต์: www.piccollege.ac.th\n"
    "พื้นที่: 13 ไร่ 70 ตร.ว.\n"
    "ผู้อำนวยการ: นางเพ็ญศรี ขุนทอง\n"
    "ปรัชญา: วินัยดี มีคุณธรรม ก้าวนำวิชาชีพ\n"
    "อัตลักษณ์: จิตสาธารณะ ทักษะเยี่ยม\n"
    "เอกลักษณ์: บริการวิชาชีพสู่ชุมชน\n"
    "คุณธรรม: มีจิตอาสา รับผิดชอบ สามัคคี มีวินัย\n"
    "วัฒนธรรม: ยิ้ม ไหว้ ทักทาย สวัสดี\n\n"

    "ผู้อำนวยการสถานศึกษา: นางเพ็ญศรี ขุนทอง\n"
    "รองผู้อำนวยการ นายสุทธิพงษ์ ศรีถัน, นายสมหมาย ศรีสุทธิ์, นายวิศาล สมโภชน์, นายภูวนาท มุสิกดิลก\n"

    "แผนกวิชา:\n"
    "ช่างยนต์\n"
    "ช่างไฟฟ้ากำลัง\n"
    "อิเล็กทรอนิกส์\n"
    "คอมพิวเตอร์ธุรกิจ\n"
    "การบัญชี\n"
    "การท่องเที่ยว\n"
    "ธุรกิจเสริมสวย\n"
    "ธุรกิจค้าปลีก\n"
    "สามัญสัมพันธ์\n"
    "เทคนิกพื้นฐาน\n\n"

    "หลักสูตรวิชาชีพระยะสั้น (ตัวอย่าง):\n"
    "เเผนกช่างยนต์:\n" 
	"ซ่อมรถ / ซ่อมรถจักรยานยนต์ / เครื่องยนต์เล็ก\n"
    "แผนกช่างไฟฟ้ากำลัง:\n" 
	"ซ่อมเครื่องใช้ไฟฟ้า / ติดตั้งไฟฟ้า / PLC พื้นฐาน\n"
    "แผนกอิเล็กทรอนิกส์:\n" 
	"ซ่อมเครื่องเสียง / ระบบเครือข่าย\n"
    "แผนกบัญชี:\n" 
	"ธุรกิจออนไลน์ / การลงทุน\n"
    "แผนกคอมพิวเตอร์:\n" 
	"ซ่อมคอม / Photoshop / Infographic\n"
    "แผนกท่องเที่ยว:\n" 
	"อาหาร-เครื่องดื่ม / มัคคุเทศก์ / MICE\n"
    "แผนกเสริมสวย:\n" 
	"ตัดผมชาย / เพนท์เล็บ\n"
    "แผนกคหกรรม:\n" 
	"ขนมอบ / ของชำร่วย / จับจีบผ้า\n"

	"อาจานย์แผนกคอมมี 5 ท่าน\n"
	"อาจารย์แผนกบัญชีมี 4 ท่าน\n"
	"อาจารย์แผนกยนต์มี 3 ท่าน\n"
	"อาจารย์แผนกไฟฟ้ากำลังมี 4 ท่าน\n"
	"อาจารย์แผนกอิเล็กทรอนิกส์มี 3 ท่าน\n"
	"อาจารย์แผนกวิชาสามัญสัมพันธ์มี 5 ท่าน\n"
	"อาจารย์แผนกวิชาเทคนิคพื้นฐานมี 2 ท่าน\n"
	"อาจารย์แผนกวิชาการท่องเที่ยวมี 2 ท่าน\n"
	"อาจารย์แผนกวิชาธุรกิจเสริมสวยมี 1 ท่าน\n"
	"อาจารย์แผนกวิชาธุรกิจค้าปลีกมี 2 ท่าน\n"
	"เจ้าหน้าที่ฝ่ายทรัพยากรมี 6 ท่าน\n"
	"เจ้าหน้าที่ฝ่ายแผนงานและความร่วมมือมี 2 ท่าน\n"
	"เจ้าหน้าที่ฝ่ายพัฒนานักเรียนนักศึกษามี 2 ท่าน\n"
	"เจ้าหน้าที่ฝ่ายวิชาการมี 7 ท่าน\n"

	"หากมีการถามถีงข้อมูลส่วนตัวของบุคลากร คุณจะแจ้งว่าไม่สามารถเผยแพร่ข้อมูลส่วนบุคคลได้ หากต้องการสอบถามเพิ่มเติมโปรดติดต่อวิทยาลัยการอาชีพปราจีนบุรี\n"
	
)
STRIP_ASTERISKS = True
HTTP_TIMEOUT = (10, 600)  # (connect, read)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=None)
CORS(app)

# ---------- Helpers ----------
def sanitize_user_prompt(text: str) -> str:
    if not text:
        return text
    return text.replace("*", "") if STRIP_ASTERISKS else text

def build_prompt(user_prompt: str) -> str:
    up = sanitize_user_prompt(user_prompt)
    return f"{SYSTEM_PROMPT}\n\nผู้ใช้: {up}\nผู้ช่วย: "

# ---------- UI routes ----------
@app.get("/")
def index():
    # เสิร์ฟ chat.html จากโฟลเดอร์เดียวกับไฟล์นี้
    return send_from_directory(BASE_DIR, "chat.html")

@app.get("/favicon.ico")
def favicon():
    # กัน 404 รบกวน log (ไม่มีรูปก็ส่ง 204 ได้)
    return ("", 204)

# ---------- API routes (proxy ไป Ollama) ----------
@app.get("/api/tags")
def api_tags():
    try:
        r = requests.get(TAGS_URL, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return Response(r.content, status=r.status_code, content_type=r.headers.get("content-type", "application/json"))
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.post("/api/generate")
def api_generate():
    data = request.get_json(force=True, silent=True) or {}
    model  = data.get("model") or DEFAULT_MODEL
    prompt = data.get("prompt") or ""
    stream = bool(data.get("stream", False))

    full_prompt = build_prompt(prompt)
    payload = {"model": model, "prompt": full_prompt, "stream": stream}

    if not stream:
        try:
            r = requests.post(GEN_URL, json=payload, timeout=HTTP_TIMEOUT, stream=False)
            r.raise_for_status()
            j = r.json()
            return jsonify({"response": j.get("response", "")})
        except Exception as e:
            return jsonify({"response": f"⚠️ เชื่อมต่อ Ollama ไม่ได้: {e}"}), 502

    def stream_gen():
        try:
            with requests.post(GEN_URL, json=payload, timeout=HTTP_TIMEOUT, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        piece = chunk.get("response", "")
                        if piece:
                            yield piece
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            yield f"\n[stream-error] {e}\n"

    return Response(stream_gen(), mimetype="text/plain; charset=utf-8")

@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5055")), debug=False)
