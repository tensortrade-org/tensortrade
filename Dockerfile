FROM python:3.6

WORKDIR /

RUN apt-get update -y && \ 
  apt-get install pandoc -y && \
  apt-get install python-mpi4py -y

RUN pip install --upgrade pip
RUN pip install -e .[tf,docs,tests,baselines,tensorforce,ccxt,fbm]
RUN pip install -r /requirements.txt
RUN pip install -r /examples/requirements.txt

