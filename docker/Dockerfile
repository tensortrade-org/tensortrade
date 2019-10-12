FROM python:3.6.9-buster
  
ADD ./requirements.txt /etc

RUN apt-get update
RUN apt-get install -y build-essential 

RUN pip install -r /etc/requirements.txt
