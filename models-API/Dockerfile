FROM python:3.9.10
WORKDIR /models_API
COPY config.py server.py models.py helper.py requirements.txt ./
COPY model model/
RUN apt-get update && apt-get install libgl1 tesseract-ocr -y
#RUN apt install tesseract-ocr
RUN pip install -r requirements.txt
RUN pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
CMD ["flask", "--app", "server", "run", "--host=0.0.0.0"]