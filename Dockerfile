FROM paperspace/fastapi-deployment:latest

WORKDIR /app

COPY main.py requirements.txt /app

RUN pip3 install -U pip && pip3 install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
