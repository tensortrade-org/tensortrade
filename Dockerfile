FROM python:3.6

WORKDIR /

COPY . ./

RUN apt-get update -y && \ 
    apt-get install pandoc -y && \
    apt-get install python-mpi4py -y

RUN pip install --upgrade pip
RUN pip install -e .[tf,docs,tests,tensorforce,ccxt,fbm]
RUN pip install -r /requirements.txt

