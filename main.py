import os
import time
import uuid
from typing import List, Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
import cohere  # 🌟 เพิ่มค่าย Cohere เข้ามา

load_dotenv()

# 🌟 โหลด API Key ทั้ง 2 ค่าย
gemini_key = os.getenv("GEMINI_API_KEY")
cohere_key = os.getenv("COHERE_API_KEY")

if not gemini_key:
    raise ValueError("ไม่พบ GEMINI_API_KEY")
if not cohere_key:
    raise ValueError("ไม่พบ COHERE_API_KEY")

# ตั้งค่า Client ทั้ง 2 ค่าย
genai.configure(api_key=gemini_key)
cohere_client = cohere.Client(api_key=cohere_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="images"), name="images")

# เก็บ session
sessions: dict[str, dict] = {}

class StartRequest(BaseModel):
    character_prompt: str
    user_prompt: str
    history: Optional[List[Dict[str, str]]] = []

class ChatRequest(BaseModel):
    session_id: str
    message: str

# 🌟 ฟังก์ชันพระเอก: ลอง Gemini ก่อน ถ้าพังให้ไป Cohere
def generate_with_fallback(system_prompt: str, history: list, message: str) -> str:
    # --- แผน A: ลองใช้ Google Gemini ---
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",  # ✏️ แก้เป็น 1.5 หรือ 2.0 (ห้ามใช้ 2.5)
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,
                max_output_tokens=1024,
            )
        )
        chat = model.start_chat(history=history)
        response = chat.send_message(message)
        print("✅ ตอบด้วย Gemini สำเร็จ")
        return response.text

    except Exception as e:
        print(f"⚠️ Gemini มีปัญหา ({e}) -> กำลังสลับไปใช้ Cohere แทน!")
        
        # --- แผน B: ถ้า Gemini พัง ให้ใช้ Cohere รับจบ ---
        try:
            # 1. ต้องแปลงประวัติแชทของ Gemini ให้เป็นฟอร์แมตของ Cohere ก่อน
            cohere_history = []
            for h in history:
                role = "USER" if h["role"] == "user" else "CHATBOT"
                text = h["parts"][0]
                cohere_history.append({"role": role, "message": text})
            
            # 2. ยิงคำถามไปหา Cohere
            response = cohere_client.chat(
                message=message,
                model="command-a-03-2025", # หรือ command-r-plus
                preamble=system_prompt,
                chat_history=cohere_history,
                temperature=0.9,
            )
            print("✅ ตอบด้วย Cohere สำเร็จ")
            return response.text
            
        except Exception as cohere_e:
            # ถ้าพังทั้ง 2 ค่าย ค่อยส่ง Error 500 กลับไปบอกหน้าเว็บ
            raise HTTPException(status_code=500, detail=f"AI ยุ่งทั้ง 2 ค่ายเลยครับ: {cohere_e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/start")
def start_session(req: StartRequest):
    session_id = str(uuid.uuid4())

    system_prompt = f"""{req.character_prompt}\n\nหมายเหตุ: ผู้ที่คุยกับคุณได้กำหนดบทบาทตัวเองไว้ดังนี้: {req.user_prompt}\nจงตอบสนองต่อบทบาทของผู้ใช้อย่างเหมาะสมตลอดเวลา"""

    formatted_history = []
    if req.history:
        for msg in req.history:
            role = "user" if msg.get("role") == "user" else "model"
            text = msg.get("text", "")
            if text:
                formatted_history.append({"role": role, "parts": [text]})

    sessions[session_id] = {
        "system": system_prompt,
        "history": formatted_history
    }
    return {"session_id": session_id}

@app.post("/chat")
def chat(req: ChatRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="ไม่พบ session กรุณาเริ่มใหม่")

    session = sessions[req.session_id]

    # เรียกใช้ระบบ Fallback
    reply = generate_with_fallback(
        system_prompt=session["system"],
        history=session["history"],
        message=req.message
    )

    # บันทึกประวัติในรูปแบบมาตรฐาน (Gemini format เป็นตัวยืน)
    session["history"].append({"role": "user", "parts": [req.message]})
    session["history"].append({"role": "model", "parts": [reply]})

    return {"reply": reply}

@app.delete("/reset/{session_id}")
def reset_session(session_id: str):
    if session_id in sessions:
        sessions[session_id]["history"] = []
    return {"status": "reset แล้ว"}
