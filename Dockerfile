FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY models.py /app/models.py
COPY server/ /app/server/

# Hugging Face Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
