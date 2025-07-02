# 경량 파이썬 베이스 이미지
FROM python:3.13-slim-bookworm

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 소스코드 복사
COPY app /app/app
COPY .env /app/.env

# FastAPI 실행 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
