import os
import io
import base64
import threading
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from gtts import gTTS

# ==============================
# 1) CONFIG
# ==============================
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", "5000"))

# URL ของ Ollama (อย่าใส่ / ท้ายสุด)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://192.168.2.37:11434")

# โมเดลเริ่มต้น (แก้ได้จาก /set-model)
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3:4b")
_model_lock = threading.Lock()

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

    "- อาคารเรียนและปฏิบัติการ 4 ชั้น จำนวน 2 หลัง\n"
    "- อาคารสำนักงาน/หอประชุม จำนวน 1 หลัง\n"
    "- บ้านพักอาศัย 7-8 จำนวน 1 หลัง\n"
    "- บ้านพักครู 6 หน่วย จำนวน 3 หลัง\n"
    "- บ้านพักภารโรง 2 หน่วย จำนวน 3 หลัง\n\n"

    "เเผนกสาขาของวิชาของวิทยาลัยการอาชีพปราจีนบุรีมี 7 สาขา:\n"
    "สาขาวิชาช่างยนต์:\n"
    "สาขาวิชาช่างไฟฟ้ากำลัง:\n"
    "สาขาวิชาอิเล็กทรอนิกส์:\n"
    "สาขาวิชาคอมพิวเตอร์ธุรกิจ:\n"
    "สาขาวิชาการบัญชี:\n"
    "สาขาวิชาการท่องเที่ยว:\n"
    "สาขาวิชาธุรกิจเสริมสวย:\n"

    "ผู้อำนวยการสถานศึกษา:\n"
    "นางเพ็ญศรี ขุนทอง ผู้อำนวยการวิทยาการอาชีพปราจีนบุรี:\n"

    "รองผู้อำนวยการสถานศึกษามี 4 ท่าน\n"
    "นายสุทธิพงษ์ ศรีถัน รองผู้อำนวยการฝ่ายบริหารทรัพยากร:\n"
    "นายสมหมาย ศรีสุทธิ์ รองผู้อำนวยการฝ่ายพัฒนากิจการนักเรียน นักศึกษา:\n"
    "นายวิศาล สมโภชน์ รองผู้อำนวยการฝ่ายวิชาการ:\n"
    "นายภูวนาท มุสิกดิลก รองฝ่ายแผนงานและความร่วมมือ:\n"

    "บุคลากร เเผนกวิชาช่างยนต์\n"
    "นายเสรี เสนชู หัวหน้าเเผนกวิชาช่างยนต์:\n"
    "นายไก่ จันทาวัน คูรเเผนกวิชาช่างยนต์:\n"
    "นายอภิรมณ์ สรรพาณิชย์ คูรเเผนกวิชาช่างยนต์:\n"

    "บุคลากร เเผนกวิชาช่างไฟฟ้ากำลัง:\n"
    "นายสมเกียรติ สุขปลั่ง หัวหน้าเเผนกช่างไฟฟ้ากำลัง:\n"
    "นายภานุมาศ บำเพ็ญชาติ คูรเเผนกวิชาไฟฟ้ากำลัง:\n"
    "นางสาวธัญญารัตน์ พูลลทรัพย์ คูรเเผนกวิชาไฟฟ้ากำลัง:\n"
    "นายชลสิทธิ์ ปัสสาราช คูรเเผนกวิชาไฟฟ้ากำลัง:\n"

    "บุคลากร เเผนกวิชาอิเล็กทรอนิกส์:\n"
    "นายปฎิณญา จันทร์สมาน หัวหน้าเเผนกวิชาช่างอิเล็กทรอนิกส์:\n"
    "นายศริพงษ์ พานทอง คูรเเผนกวิชาช่างอิเล็กทรอนิกส์:\n"
    "นางสาวอัจฉรา อยู่อินทร์ คูรเเผนกวิชาช่างอิเล็กทรอนิกส์:\n"

    "บุคลากร เเผนกวิชาการบัญชี:\n"
    "นางนิภา งามเเสง หัวหน้าเเผนกวิชาการบัญชี:\n"
    "นางสาวณัฐพัชร์ สิทธิพันธ์ คูรเเผนกวิชาการบัญชี:\n"
    "นางสาวนฤมล จันทรา คูรเเผนกวิชาการบัญชี:\n"
    "นางสาวอริศรา พรมโคตร คูรเเผนกวิชาการบัญชี:\n"

    "บุคลากร เเผนกคอมพิวเตอร์ธุรกิจ:\n"
    "นายภูวนาท มุสิกดิลก หัวหน้าเเผนกคอมพิวเตอร์ธุรกิจ:\n"
    "นายวีระชัย ปิ่นคำ คูรเเผนกวิชาคอมพิวเตอร์ธุรกิจ:\n"
    "นางสาวจุทารัตน์ คงประสิทธิ์ คูรเเผนกวิชาคอมพิวเตอร์ธุรกิจ:\n"
    "นางพิไลวรรณ ตุ่นทอง คูรเเผนกวิชาคอมพิวเตอร์ธุรกิจ:\n"
    "นางสุภาพร ศรีถัน คูรเเผนกวิชาคอมพิวเตอร์ธุรกิจ:\n"

    "บุคลากร เเผนกวิชาสามัญ-สัมพันธ์:\n"
    "นางนุชอนงค์ คงเทศ หัวหน้าเเผนกวิชาสามัญ-สัมพันธ์:\n"
    "นางชวิดา ศุภคุณกิตติ คูรเเผนกวิชาสามัญ-สัมพันธ์:\n"
    "นายภาคภูมิ นาคทิพย์ คูรเเผนกวิชาสามัญ-สัมพันธ์:\n"
    "นางสาวกรองเเก้ว โฉมงาม คูรเเผนกวิชาสามัญ-สัมพันธ์:\n"
    "นายกฤษดา ถวิลรักษ์ คูรเเผนกวิชาสามัญ-สัมพันธ์:\n"

    "บุคลากร เเผนกวิชาเทคนิกพื้นฐาน:\n"
    "นายวันชัย พันธุมงคล หัวหน้าเเผนกวิชาเทคนิกพื้นฐาน:\n"
    "นายหาญประชา พรหมมา คูรเเผนกวิชาเทคนิกพื้นฐาน:\n"

    "บุคลากร เเผนกวิชาการท่องเที่ยว:\n"
    "นางสาวอรวรรณ นิยมสิทธิ์ หัวหน้าเเผนกวิชาการท่องเที่ยว:\n"
    "นางสาวพรวิมล ไมตรีจิตต์ คูรแผนกวิชาการท่องเที่ยว:\n"

    "บุคลากร เเผนกวิชาธุรกิจเสริมสวย:\n"
    "นางสาวสุภาพร ดางาม หัวหน้าแผนกวิชาธุรกิจเสริมสวย:\n"

    "บุคลากร แผนกวิชาธุรกิจค้าปลีก:\n"
    "นางฐานิต โภคทรัพย์ หัวหน้าแผนกวิชาธุรกิจค้าปลีก:\n"

    "หลักสูตรวิชาชีพระยะสั้น (ตัวอย่าง):\n"
    "เเผนกช่างยนต์: ซ่อมรถ / ซ่อมรถจักรยานยนต์ / เครื่องยนต์เล็ก\n"
    "แผนกช่างไฟฟ้ากำลัง: ซ่อมเครื่องใช้ไฟฟ้า / ติดตั้งไฟฟ้า / PLC พื้นฐาน\n"
    "แผนกอิเล็กทรอนิกส์: ซ่อมเครื่องเสียง / ระบบเครือข่าย\n"
    "แผนกบัญชี: ธุรกิจออนไลน์ / การลงทุน\n"
    "แผนกคอมพิวเตอร์: ซ่อมคอม / Photoshop / Infographic\n"
    "แผนกท่องเที่ยว: อาหาร-เครื่องดื่ม / มัคคุเทศก์ / MICE\n"
    "แผนกเสริมสวย: ตัดผมชาย / เพนท์เล็บ\n"
    "แผนกคหกรรม: ขนมอบ / ของชำร่วย / จับจีบผ้า\n"

    "สาขาวิชาช่างไฟฟ้า:\n"
    "งานซ่อมเครื่องใช้ไฟฟ้าภายในบ้าน:\n"
    "PLC พื้นฐาน:\n"
    "งานติดตั้งไฟฟ้าภายในอาคารและอื่นๆ:\n"

    "สาขาวิชาช่างอิเล็กทรอนิกส์:\n"
    "อิเล็กทรอนิกส์พื้นฐาน:\n"
    "งานอซ่อมเครื่องเล่นเสียงและภาพระบบดิจิตอล (VCD/DVD):\n"
    "งานติดตั้งระบบเครือข่ายคอมพิวเตอร์และอื่นๆ:\n"

    "สาขาวิชาบัญชี:\n"
    "เริ่มต้นธุระกิจอย่างมีทิศทาง:\n"
    "การขายสินค้าออนไลน์:\n"
    "เทคนิคการลงทุนสำหรับนักลงทุนรายใหม่และอื่นๆ:\n"

    "สาขาวิชาคอมพิวเตอร์:\n"
    "งานซ่อมบำรุงรักษาคอมพิวเตอร์:\n"
    "เทคนิคการใช้โปรแกรม Adobd Photoshop:\n"
    "การออกแบบอินโฟกราฟิกส์เบื้องต้นและอื่นๆ:\n"

    "สาขาวิขาการท่องเที่ยว:\n"
    "อาหารและเครื่องดื่มเพื่อการท่องเที่ยว:\n"
    "มัคคุเทศก์:\n"
    "การจัดการธุรกิจ MICEและอื่นๆ:\n"

    "สาขาวิชางานเสริมสวย:\n"
    "พื้นฐานการตัดผมชาย:\n"
    "ช่างเสริมสวยเบื้องต้น:\n"
    "แต่งเล็บ-เพ็นท์เล็บและอื่นๆ:\n"

    "แผนกวิชาคหกรรม(ระยะสั้น):\n"
    "ขนมอบ:\n"
    "ขนมไทย:\n"
    "ประดิษฐ์ของชำรวยและของที่ระลึก:\n"
    "จับจีบและผูกผ้าและอื่นๆ:\n"

    "และมีวิชาอื่นๆอีกมากมายสามรถติดต่อได้ที่วิทยาลัยการอาชีพปราจีนบุรี 037-212220 และเว็บไซต์ของวิทยาค่ะ:\n"

    "หากมีการถามถีงข้อมูลส่วนตัวของบุคลากร คุณจะแจ้งว่าไม่สามารถเผยแพร่ข้อมูลส่วนบุคคลได้ หากต้องการสอบถามเพิ่มเติมโปรดติดต่อวิทยาลัยการอาชีพปราจีนบุรี\n"
)


# ==============================
# 1.1) DATA: อาจารย์
# ==============================
PROFESSOR_DATA = {
    "อาจารย์วีระชัย": {
        "aliases": ["อาจารย์วีระชัย", "อาจารย์วี"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/06/24.jpg",
        "title": "นาย วีระชัย ปิ่นคำ"
    },
    "อาจารย์ภูวนาท": {
        "aliases": ["อาจารย์ภูวนาท", "อาจารย์ปิ๊ก"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/06/7.jpg",
        "title": "นาย ภูวนาท มุสิกดิลก"
    },
    "อาจารย์จุทารัตน์": {
        "aliases": ["อาจารย์จุทารัตน์", "อาจารย์อ้อม"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/06/38.jpg",
        "title": "นางสาว จุทารัตน์ คงประสิทธิ์"
    },
    "อาจารย์สุภาพร_ศรีถัน": {
        "aliases": ["อาจารย์สุภาพร", "อาจารย์เอ๋"],
        "image_url": "",
        "title": "นาง สุภาพร ศรีถัน"
    },
    "อาจารย์พิไลวรรณ": {
        "aliases": ["อาจารย์พิไลวรรณ", "อาจารย์โอ๋"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/06/49.jpg",
        "title": "นาง พิไลวรรณ ตุ่นทอง"
    },
    "อาจารย์เสรี": {
        "aliases": ["อาจารย์เสรี"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/06/11.jpg",
        "title": "นาย เสรี เสนชู"
    },
    "อาจารย์ไก่": {
        "aliases": ["อาจารย์ไก่"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/06/26.jpg",
        "title": "นาย ไก่ จันทาวัน"
    },
    "อาจารย์อภิรมณ์": {
        "aliases": ["อาจารย์อภิรมณ์", "อาจารย์โต้ง"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/06/40.jpg",
        "title": "นาย อภิรมณ์ สรรพาณิชย์"
    },
    "อาจารย์สมเกียรติ": {
        "aliases": ["อาจารย์สมเกียรติ", "อาจารย์โก้"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/06/40.jpg",
        "title": "นาย สมเกียรติ สุขปลั่ง"
    },
    "อาจารย์ภานุมาศ": {
        "aliases": ["อาจารย์ภานุมาศ", "อาจารย์อาร์ต"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/06/18.jpg",
        "title": "นาย ภานุมาศ บำเพ็ญชาติ"
    },
    "อาจารย์ธัญญารัตน์": {
        "aliases": ["อาจารย์ธัญญารัตน์", "อาจารย์ยุ้ย"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/06/28.jpg",
        "title": "นางสาว ธัญญารัตน์ พูลลทรัพย์"
    },
    "ผู้อำนวยการเพ็ญศรี": {
        "aliases": ["เพ็ญศรี", "ขุนทอง"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/08/2.png",
        "title": "นาง เพ็ญศรี ขุนทอง"
    },
    "รองสุทธิพงษ์": {
        "aliases": ["สุทธิพงษ์", "ศรีถัน"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/08/3.png",
        "title": "นาย สุทธิพงษ์ ศรีถัน"
    },
    "รองสมหมาย": {
        "aliases": ["สมหมาย", "ศรีสุทธิ์"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/08/5.png",
        "title": "นาย สมหมาย ศรีสุทธิ์"
    },
    "รองวิศาล": {
        "aliases": ["วิศาล", "สมโภชน์"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/08/4.png",
        "title": "นาย วิศาล สมโภชน์"
    },
    "รองภูวนาท": {
        "aliases": ["ภูวนาท", "มุสิกดิลก"],
        "image_url": "https://www.piccollege.ac.th/wp-content/uploads/2025/08/6.png",
        "title": "นาย ภูวนาท มุสิกดิลก"
    },
}

dev_log = []  # เก็บประวัติการสนทนาแบบเบา ๆ

# ==============================
# 2) APP INIT
# ==============================
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # เปิด CORS ให้เรียกจาก front ได้สะดวก

# ==============================
# 3) HELPERS
# ==============================
def current_model() -> str:
    with _model_lock:
        return MODEL_NAME

def set_current_model(name: str):
    global MODEL_NAME
    with _model_lock:
        MODEL_NAME = name

def ping_ollama_ok() -> bool:
    # ใช้ /api/version เพราะ / อาจถูกห้าม
    try:
        r = requests.get(f"{OLLAMA_API_URL}/api/version", timeout=3)
        return r.ok
    except Exception:
        return False

def get_ollama_response(user_input: str, system_prompt: str) -> str:
    """เรียก Ollama (OpenAI-compatible /v1/chat/completions)"""
    try:
        payload = {
            "model": current_model(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        }
        r = requests.post(f"{OLLAMA_API_URL}/v1/chat/completions", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        if not text:
            raise ValueError("ไม่มีข้อความตอบกลับจาก Ollama")
        dev_log.append({"role": "ai", "content": text})
        return text
    except Exception as e:
        msg = f"⚠️ Ollama Error: {e}"
        dev_log.append({"role": "error", "content": msg})
        return "ขออภัยค่ะ มีปัญหาในการเชื่อมต่อกับ AI ค่ะ"

def detect_professor_from_input(user_input: str):
    s = (user_input or "").lower()
    hits = []
    for name, data in PROFESSOR_DATA.items():
        aliases = [name.lower()] + [a.lower() for a in data.get("aliases", [])]
        if any(a in s for a in aliases):
            hits.append((name, data))
    return hits  # [(ชื่อ, data)]

def generate_audio_base64(text: str):
    try:
        tts = gTTS(text=text, lang='th', slow=False, tld='com')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return base64.b64encode(mp3_fp.read()).decode('utf-8')
    except Exception as e:
        dev_log.append({"role": "error", "content": f"TTS Error: {e}"})
        return None

# ==============================
# 4) ROUTES
# ==============================
@app.get("/health")
def health():
    ok = ping_ollama_ok()
    return jsonify({"ok": ok, "model": current_model()})

@app.get("/models")
def list_models():
    """ดึงรายชื่อโมเดลจาก Ollama: GET /api/tags"""
    try:
        r = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        names = [it.get("name") for it in data.get("models", []) if it.get("name")]
        return jsonify({"models": names})
    except Exception as e:
        return jsonify({"error": f"fetch models failed: {e}"}), 502

@app.post("/set-model")
def set_model():
    """เปลี่ยนโมเดลเริ่มต้น (global)"""
    j = request.get_json(silent=True) or {}
    m = (j.get("model") or "").strip()
    if not m:
        return jsonify({"error": "missing model"}), 400
    set_current_model(m)
    return jsonify({"ok": True, "model": current_model()})

@app.get("/history")
def history():
    """ส่ง dev_log ล่าสุด (สูงสุด 50 รายการ)"""
    return jsonify({"items": dev_log[-50:]})

@app.route("/")
def index():
    # ชี้ไปที่ templates/index.html (หน้า chat)
    # ถ้าไม่มีไฟล์ ให้เปลี่ยนเป็นส่ง string หรือส่งไฟล์ที่คุณตั้งไว้
    try:
        return render_template("index.html")
    except:
        return "Flask is running. Put your chat UI in templates/index.html", 200

@app.post("/ask")
def ask_ai():
    try:
        data = request.get_json(force=True)
        user_text = (data or {}).get('text', '').strip()
        if not user_text:
            return jsonify({"error": "ไม่พบข้อความในคำขอ"}), 400

        dev_log.append({"role": "user", "content": user_text})

        # ตรวจชื่ออาจารย์
        found_profs = detect_professor_from_input(user_text)
        image_urls = []
        prof_name = None
        prof_title = None

        if found_profs:
            prof_name, pdata = found_profs[0]
            prof_title = pdata.get("title", prof_name)

            for _, d in found_profs:
                url = d.get("image_url")
                if isinstance(url, str) and url:
                    image_urls.append(url)
                elif isinstance(url, list):
                    image_urls.extend([u for u in url if u])

            image_urls = list(dict.fromkeys(image_urls))  # unique
            prof_list_text = " และ ".join([p[1].get("title", p[0]) for p in found_profs])

            ai_response_text = (
                f"ฉันได้แสดงรูปภาพของอาจารย์แล้วค่ะ หรือไม่คะ? "
                f"คุณพูดถึง {prof_list_text} มีอะไรให้ช่วยอีกไหมคะ"
            )
        else:
            ai_response_text = get_ollama_response(user_text, SYSTEM_PROMPT)

        # สร้างเสียงตอบกลับ (base64)
        audio_b64 = generate_audio_base64(ai_response_text)
        audio_url = f"data:audio/mp3;base64,{audio_b64}" if audio_b64 else None

        single_image = image_urls[0] if image_urls else None

        return jsonify({
            "status": "success",
            "text_response": ai_response_text,
            "audio_url": audio_url,
            "image_url": single_image,
            "image_urls": image_urls,
            "professor_name": prof_name,
            "professor_title": prof_title
        })
    except Exception as e:
        msg = f"เกิดข้อผิดพลาด: {e}"
        dev_log.append({"role": "error", "content": msg})
        return jsonify({"error": msg}), 500

# ==============================
# 5) MAIN
# ==============================
if __name__ == "__main__":
    print(f"Server starting on http://{APP_HOST}:{APP_PORT}  (Ollama: {OLLAMA_API_URL}, model: {MODEL_NAME})")
    app.run(host=APP_HOST, port=APP_PORT, debug=True, threaded=True)
