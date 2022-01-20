# FROM ubuntu:latest
#
# https://github.com/h5py/h5py/issues/1461
# underlying issue is that you do not have hdf5 installed on your system.
#
# RUN apt-get update -y
# RUN apt-get install -y tzdata
# ENV TZ America/New_York
#
# RUN apt-get -y install build-essential git liblapacke-dev libblas-dev libhdf5-103 libhdf5-dev python3 python3-h5py python3-pip python3-scipy tesseract-ocr

# RUN apt-file update
# RUN apk add --no-cache cmake extra-cmake-modules build-base git opencv opencv-dev python3 python3-dev py3-pip tesseract-ocr

# ARG USER=idle
# RUN adduser --disabled-password --gecos "" $USER
# RUN echo "$USER:$USER" | chpasswd

# RUN addgroup $USER \
#  && adduser -D -s /bin/sh -G $USER $USER \
#  && echo "$USER:$USER" | chpasswd
 
# USER $USER
# WORKDIR /home/$USER

# RUN echo PATH=/home/idle/.local/bin:"$PATH" >> /home/idle/.bashrc
# RUN echo "TZ=America/New_York" >> /home/idle/.bashrc
# RUN pip install --upgrade pip scipy h5py
# RUN pip install opencv-python
# imageai pytesseract

# RUN git clone https://github.com/gautada/coin-vision.git

# ╭――――――――――――――――---------------------------------------------------------――╮
# │                                                                           │
# │ Tensorflow Build                                                          │
# │                                                                           │
# │ https://github.com/gautada/coin-vision                                    │
# │                                                                           │
# │ This project uses tensorflow and a coin sorter to identify and sort       │
# │ pennies based on their visibile information.                              │
# │                                                                           │
# ╰―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――╯

