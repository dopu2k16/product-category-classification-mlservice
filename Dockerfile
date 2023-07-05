FROM ubuntu:latest

MAINTAINER Mitodru Niyogi <mitodru.niyogi@gmail.com>

RUN apt-get update -y

RUN apt-get install -y python3-pip python3-dev build-essential

COPY . /MLE-challenge-Gfk

WORKDIR /MLE-challenge-Gfk

RUN pip3 install -r requirements.txt

ENTRYPOINT [“python3”]

CMD [“app.py”]
