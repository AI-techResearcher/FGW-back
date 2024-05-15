FROM python:3.8
RUN python3 -m pip install flask
WORKDIR /app
COPY ./requirements.txt .
ENV ENVIRONMENT=production
RUN pip install -r ./requirements.txt