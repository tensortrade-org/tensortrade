FROM python:3.5

WORKDIR /

COPY . ./

RUN pip install -e .[tf,docs,tests,tensorforce,ccxt,fbm]
