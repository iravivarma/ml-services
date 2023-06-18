FROM python:3.7.17-slim-buster

WORKDIR /app

RUN apt-get update

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "Main.py", "--server.port=8502", "--server.address=0.0.0.0" ]