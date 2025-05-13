FROM python:3.12

WORKDIR /car-price

ADD requirements.txt .

RUN pip install -r requirements.txt

ADD . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]