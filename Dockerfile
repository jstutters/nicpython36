
ARG cuda_version=9.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel
MAINTAINER thisgithub

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

# Install Python packages and keras
ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown $NB_USER $CONDA_DIR -R && \
    mkdir -p /src && \
    chown $NB_USER /src

# USER $NB_USER
USER root
ARG python_version=3.6

RUN conda config --append channels conda-forge

        
# Install git, wget, python-dev, pip, BLAS + LAPACK and other dependencies
RUN apt-get update && apt-get install -y \
  gfortran \
  liblapack-dev \
  libopenblas-dev \
  python-dev \
  python-tk\
  git \
  curl \
  emacs24
  
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*        


    
ENV PATH /opt/conda/bin:$PATH
ENV PATH /opt/conda/envs/idp/bin:$PATH


RUN conda update conda
RUN conda update anaconda
RUN conda update --all
RUN conda config --add channels intel
RUN conda create -n idp intelpython3_full python=3
# RUN echo "source activate idp" > ~/.bashrc
# RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Install miniconda to /miniconda
# RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
# RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
# RUN rm /Miniconda-latest-Linux-x86_64.sh
# ENV PATH=/opt/conda/bin:${PATH}

# ENV PATH=/miniconda/envs/idp/bin:$PATH
# RUN conda remove -n tensorflow
ARG python_version=3.6

RUN conda config --append channels conda-forge
RUN conda install -y python=${python_version} && \
    # pip install --upgrade pip && \
    pip install \
      sklearn_pandas \
      tensorflow-gpu \
      cntk-gpu && \
    conda install \
      numpy \
      scipy \
      bcolz \
      h5py \
      matplotlib \
      mkl \
      nose \
      notebook \
      Pillow \
      pandas \
      pydot \
      pygpu \
      pyyaml \
      scikit-learn \
      six \
      theano \
      pygpu \
      mkdocs \
      && \
    git clone git://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
    pip install git+git://github.com/keras-team/keras.git && \
    conda clean -yt
# install CNN related packages
ADD requirements.txt /requirements.txt
# RUN conda install numpy scipy mkl
# RUN conda install theano pygpu
# RUN pip install pip --upgrade
RUN pip install -r /requirements.txt
# RUN pip uninstall protobuf
# RUN conda install tensorflow-gpu

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
