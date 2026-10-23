FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY train_model.py ml_service.py /app/

EXPOSE 5000
CMD ["sleep", "infinity"]
