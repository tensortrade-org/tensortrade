FROM continuumio/miniconda3

RUN apt update && \
    apt install python3 python3-pip git wget -y

RUN git clone https://github.com/robertalanm/tensortrade.git


#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#    bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/conda && \
#    rm Miniconda3-latest-Linux-x86_64.sh && \
#    export PATH=~/conda/bin:$PATH && \
#    . ~/.bashrc

WORKDIR tensortrade

RUN [ "conda", "env", "create" ]

RUN [ "/bin/bash", "-c", "source activate tensortrade" ]

RUN pip3 install -r requirements.txt

#RUN pip3 install runipy

#RUN runipy examples/TensorTrade_Tutorial.ipynb

RUN jupyter notebook
