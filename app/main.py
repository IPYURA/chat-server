import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
import supabase
import openai
import srt
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging
import numpy as np
import ast
import re

#  환경변수 로드
load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "https://cdg-chatbot-practice.vercel.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase 초기화
supabase_client = supabase.create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# OpenAI 클라이언트 (비동기 대응을 위해 AsyncOpenAI 사용)
from openai import AsyncOpenAI
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# SRT 처리 함수
async def process_srt_file(file: UploadFile) -> List[srt.Subtitle]:
    try:
        contents = await file.read()
        return list(srt.parse(contents.decode('utf-8-sig')))  # BOM 처리
    except Exception as e:
        logger.error(f"SRT 파싱 실패: {e}")
        raise HTTPException(status_code=400, detail=f"SRT 파싱 오류: {str(e)}")

# 임베딩 생성 함수 (비동기)
async def generate_embedding(text: str) -> List[float]:
    try:
        response = await openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"임베딩 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 오류: {str(e)}")

# 자막 업로드 엔드포인트
@app.post("/upload-subtitle")
async def upload_srt_endpoint(subtitle_file: UploadFile = File(...), video_id: str = Form(...)):
    try:
        subtitles = await process_srt_file(subtitle_file)
        
        for sub in subtitles:
            time_text = f"Start: {sub.start.total_seconds()}, End: {sub.end.total_seconds()}, Text: {sub.content}"
            embedding = await generate_embedding(time_text)
            
            supabase_client.table('subtitle_chunks').insert({
                "vector": embedding,
                "start_time": sub.start.total_seconds(),
                "end_time": sub.end.total_seconds(),
                "video_id": video_id,
                "raw_text": sub.content,
                "created_at": datetime.now().isoformat()
            }).execute()
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    prompt: str
    video_id: str

# 시간 정보 추출 함수
def extract_seconds(prompt: str):
    # "1분30초" 또는 "90초" 등 다양한 케이스 파싱
    m = re.search(r'(\d+)\s*분\s*(\d+)?\s*초?', prompt)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2) or 0)
        return minutes * 60 + seconds
    m = re.search(r'(\d+)\s*초', prompt)
    if m:
        return int(m.group(1))
    return None

# 코사인 유사도 함수
def cosine_similarity(a: List[float], b) -> float:
    # b가 문자열인 경우 파싱
    if isinstance(b, str):
        b = ast.literal_eval(b)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 프롬프트에서 시간 정보 추출
        seconds = extract_seconds(request.prompt)
        
        # 해당 video_id의 subtitle_chunks 모두 조회
        response = supabase_client.table('subtitle_chunks') \
            .select("id, vector, raw_text, start_time, end_time") \
            .eq("video_id", request.video_id) \
            .execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="해당 video_id의 자막 데이터가 없습니다.")
        
        context = ""
        # 시간 정보가 있으면 해당 시간에 포함되는 자막 추출
        if seconds is not None:
            target_chunks = [
                chunk for chunk in response.data
                if chunk['start_time'] <= seconds <= chunk['end_time']
            ]
            if target_chunks:
                context = "\n".join([chunk['raw_text'] for chunk in target_chunks])
        
        # 시간 정보가 없거나 해당 시간에 자막이 없으면 임베딩 기반 검색
        if not context:
            prompt_embedding = await generate_embedding(request.prompt)
            for chunk in response.data:
                # vector 문자열을 실제 배열로 변환
                if isinstance(chunk['vector'], str):
                    vector = np.array(ast.literal_eval(chunk['vector']), dtype=float)
                else:
                    vector = np.array(chunk['vector'], dtype=float)
                chunk['similarity'] = cosine_similarity(prompt_embedding, vector)
            # 유사도 상위 3개 추출
            top_chunks = sorted(response.data, key=lambda x: x['similarity'], reverse=True)[:3]
            context = "\n".join([chunk['raw_text'] for chunk in top_chunks])

        # GPT로 답변 생성
        gpt_response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "영상 컨텍스트 기반 질문 응답"},
                {"role": "user", "content": f"Context: {context}\nQuestion: {request.prompt}"}
            ]
        )

        return {"response": gpt_response.choices[0].message.content}
    
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 테스트 엔드포인트
@app.get('/test')
async def test():
    return {"res": "this is test response~"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
