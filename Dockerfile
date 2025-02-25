FROM ubuntu:latest

WORKDIR /ml
COPY ./ml /ml/

RUN git clone https://github.com/hyperledger/fabric-samples
RUN git clone https://github.com/hyperledger-caliper/caliper-benchmarks
RUN cd ml
RUN pip install -r requirments.txt


CMD [ "python3", "main.py" ]