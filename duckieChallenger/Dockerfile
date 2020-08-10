# Definition of Submission container

# We start from tensorflow-gpu image
FROM tensorflow/tensorflow:2.1.0-gpu-py3

# this line prevents interaction with tzdata package
ENV DEBIAN_FRONTEND=noninteractive
# from https://github.com/tensorflow/tensorflow/issues/10776
# this may not be needed if you use pytorch image
RUN cd /usr/local/cuda/lib64 \
    && mv stubs/libcuda.so ./ \
    && ln -s libcuda.so libcuda.so.1 \
    && ldconfig

# DO NOT MODIFY: your submission won't run if you do
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         gcc \
         libsm6\
         libxext6\
         libxrender-dev\
         libc-dev\
         git \
         bzip2 \
         python-tk && \
     rm -rf /var/lib/apt/lists/*

# let's create our workspace, we don't want to clutter the container
RUN mkdir /workspace

# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN pip uninstall -y keras
RUN pip install keras --upgrade


# let's copy all our solution files to our workspace
# if you have more file use the COPY command to move them to the workspace
COPY solution.py /workspace
#COPY BestAIDO3FrankNet.h5 /workspace
COPY FrankNet.h5 /workspace
COPY helperFncs.py /workspace
# we make the workspace our working directory
WORKDIR /workspace

# DO NOT MODIFY: your submission won't run if you do
ENV DUCKIETOWN_SERVER=evaluator

# let's see what you've got there...
CMD python3 solution.py
