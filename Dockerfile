FROM ubuntu:bionic
RUN apt-get update
RUN apt-get install -y git cmake g++ libboost-all-dev libblas-dev liblapack-dev libeigen3-dev
RUN apt-get install -y python-dev python3-pip
RUN pip3 install pybind11 sphinx numpy
RUN mkdir /work
VOLUME /work/code
WORKDIR /work