FROM python:3.9-slim

WORKDIR /opt/src

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt


COPY . .

ENTRYPOINT ["python3", "main.py"]