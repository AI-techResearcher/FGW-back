FROM base
COPY . .
CMD ["python", "ChatAdvanced.py"]

# docker build -t fgw .
# docker run -d --name fgw-flask-app -p 5000:5000 fgw