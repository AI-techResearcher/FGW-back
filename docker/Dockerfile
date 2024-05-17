FROM python:3.8
RUN python3 -m pip install flask
WORKDIR /app
COPY . .
RUN pip install -r ./requirements.txt
ENV ENVIRONMENT=production
CMD ["python", "ChatBasic.py"]

# docker build -t fgw .
# docker run -d --name fgw-flask-app -p 5000:5000 fgw