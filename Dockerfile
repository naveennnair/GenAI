FROM python:3.8-alpine

WORKDIR /App

COPY . /App

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]