FROM paperspace/fastapi-deployment:latest

WORKDIR D:/Pritesh/Datacore_Consultants/Projects/HuggingFace/Deliverables/AI_Image_Detector/app

COPY main.py requirements.txt

RUN pip3 install -U pip && pip3 install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
