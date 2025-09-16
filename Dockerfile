FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY weights/ weights/
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
