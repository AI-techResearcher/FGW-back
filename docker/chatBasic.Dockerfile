FROM base
COPY . .
CMD ["python", "ChatBasic.py"]

# docker build -t fgw .
# docker run -d --name fgw-flask-app -p 5000:5000 fgw