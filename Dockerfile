FROM python:3.6

WORKDIR /

COPY . ./

RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz &&\
  tar xzvf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib && \
  ./configure && \
  make && \
  make install && \
  cd .. && \
  rm -rf ta-lib*

RUN apt-get update -y && \ 
  apt-get install pandoc -y && \
  apt-get install python-mpi4py -y

RUN pip install --upgrade pip
RUN pip install -e .[tf,docs,tests,baselines,tensorforce,ccxt,fbm]
RUN pip install -r ./requirements.txt
RUN pip install -r ./examples/requirements.txt