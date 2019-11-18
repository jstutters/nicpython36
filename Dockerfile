FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER thisgithub

# Install git, wget, python-dev, pip, BLAS + LAPACK and other dependencies
RUN apt-get update && apt-get install -y \
  gfortran \
  git \
  wget \
  liblapack-dev \
  libopenblas-dev \
  python-dev \
  python-tk\
  git \
  curl \
  emacs24
ENV PATH /opt/conda/bin:$PATH
ENV PATH /opt/conda/envs/idp/bin:$PATH

# Add conda environment files (.yml)
COPY ["./conda_environments/", "."]

USER root
ENV CUDA_ROOT /usr/local/cuda/bin
# Get installation file
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh -O ~/anaconda.sh

# Install anaconda at /opt/conda
RUN /bin/bash ~/anaconda.sh -b -p "/opt/conda"

# Remove installation file
RUN rm ~/anaconda.sh

# Make conda command available to all users
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
# Activate conda environment with interactive bash session
RUN conda update -y conda
RUN conda config --add channels intel
RUN conda create -n idp intelpython3_full python=3
RUN echo "source activate idp" > ~/.bashrc
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Install miniconda to /miniconda
# RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
# RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
# RUN rm /Miniconda-latest-Linux-x86_64.sh
# ENV PATH=/opt/conda/bin:${PATH}

# ENV PATH=/miniconda/envs/idp/bin:$PATH
# RUN conda remove -n tensorflow

# install CNN related packages
ADD requirements.txt /requirements.txt
RUN conda install numpy scipy mkl
RUN conda install theano pygpu
RUN pip install pip --upgrade
RUN pip install -r /requirements.txt
# RUN pip uninstall protobuf
RUN pip install --upgrade tensorflow-gpu

# create a docker user
RUN useradd -ms /bin/bash docker
ENV HOME /home/docker

# copy necessary files to container
RUN mkdir $HOME/src
ENV PATH=/$HOME/src:${PATH}
ADD __init__.py $HOME/src/
ADD .theanorc $HOME/src/
# ADD .keras $HOME/src/
# RUN mkdir $HOME/src/.theanorc
# ENV PATH=/$HOME/src/.theanorc:${PATH}
# ADD .theanorc $HOME/src/.theanorc/
# RUN mkdir $HOME/src/.keras
# ENV PATH=/$HOME/src/.keras:${PATH}
# ADD .keras $HOME/src/.keras/
ADD app.py $HOME/src/
ADD cnn_scripts.py $HOME/src/
# ADD config $HOME/src/config
# ADD nets $HOME/src/nets
ADD libs $HOME/src/libs
ADD utils $HOME/src/utils
ADD logonic.png $HOME/src/
ADD nic_train_network_batch.py $HOME/src/
ADD nic_infer_segmentation_batch.py $HOME/src/
ADD tensorboardlogs $HOME/src/
# add permissions (odd)
# RUN chown docker -R nets
# RUN chown docker -R config

USER docker
WORKDIR $HOME/src
