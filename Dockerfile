FROM python:3.6

WORKDIR /tensforforce


COPY . ./


RUN apt-get update -y && \ 
  apt-get install pandoc -y && \
  apt-get install python-mpi4py -y




RUN pip install --upgrade pip
RUN pip3 install .[tf,docs,tests,baselines,tensorforce,ccxt,fbm]
RUN ls -la
RUN pip3 install -r ./requirements.txt
RUN pip3 install -r ./examples/requirements.txt
RUN pip3 install mpi4py