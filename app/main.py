# 서버 켜지는것까지 확인됨.

# main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import supabase
import openai
import srt
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000, http://localhost:3001, http://localhost:3002, https://cdg-chatbot-practice.vercel.app/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase 초기화
supabase_client = supabase.create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# OpenAI 초기화
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pydantic 모델
class ChatRequest(BaseModel):
    prompt: str
    video_id: str

# SRT 처리 함수
async def process_srt_file(file: UploadFile):
    try:
        contents = await file.read()
        return list(srt.parse(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SRT 파싱 오류: {str(e)}")

# 임베딩 생성 함수
def generate_embedding(text: str):
    return openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    ).data[0].embedding

# 라우터
@app.post("/upload-srt")
async def upload_srt_endpoint(file: UploadFile = File(...), video_id: str = None):
    subtitles = await process_srt_file(file)
    
    for sub in subtitles:
        time_text = f"Start: {sub.start.total_seconds()}, End: {sub.end.total_seconds()}, Text: {sub.content}"
        embedding = generate_embedding(time_text)
        
        supabase_client.table('subtitle_chunks').insert({
            "vector": embedding,
            "start_time": sub.start.total_seconds(),
            "end_time": sub.end.total_seconds(),
            "video_id": video_id,
            "raw_text": sub.content,
            "created_at": datetime.now().isoformat()
        }).execute()
    
    return {"status": "success"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    prompt_embedding = generate_embedding(request.prompt)
    
    response = supabase_client.rpc('match_subtitle_chunks', {
        'query_embedding': prompt_embedding,
        'match_threshold': 0.7,
        'match_count': 3,
        'video_id': request.video_id
    }).execute()
    
    context = "\n".join([item['raw_text'] for item in response.data])
    gpt_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "영상 컨텍스트 기반 질문 응답"},
            {"role": "user", "content": f"Context: {context}\nQuestion: {request.prompt}"}
        ]
    )
    
    return {"response": gpt_response.choices[0].message.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
