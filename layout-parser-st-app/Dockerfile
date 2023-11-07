FROM python:3.9.10
WORKDIR /layout-parser-app
COPY app.py helper.py requirements.txt ./
COPY example_image example_image/
RUN apt-get update && apt-get install libgl1 -y
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "./app.py"]