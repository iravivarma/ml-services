
FROM python:3.9-slim-bullseye

WORKDIR /app

RUN apt-get update

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#EXPOSE 8080

CMD ["uvicorn","routes:app","--host", "0.0.0.0", "--port", "80","--reload"]