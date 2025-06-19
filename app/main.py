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

# 환경변수 로드
load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "https://cdg-chatbot-practice.vercel.app/"
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

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        prompt_embedding = await generate_embedding(request.prompt)
        
        response = supabase_client.rpc('match_subtitle_chunks', {
            'query_embedding': prompt_embedding,
            'match_threshold': 0.7,
            'match_count': 3,
            'target_video_id': request.video_id  # 파라미터명 변경
        }).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="No matching data found")
        
        context = "\n".join([item['raw_text'] for item in response.data])
        
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
