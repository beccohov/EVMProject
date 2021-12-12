FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y pip
COPY . /app
WORKDIR /app 
RUN apt-get update && apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y build-essential

RUN pip install -r requirements.txt
ENTRYPOINT [ "python3" ]

CMD ["src/app.py"]