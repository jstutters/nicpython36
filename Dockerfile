ARG UBUNTU_VERSION=18.04
MAINTAINER thisgithub

ARG ARCH=
ARG CUDA=10.0
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.2.24-1

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        cuda-cublas-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip
        
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

RUN [ ${ARCH} = ppc64le ] || (apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*)

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

ARG USE_PYTHON_3_NOT_2
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow-gpu
ARG TF_PACKAGE_VERSION=
RUN ${PIP} install ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc


    
ENV PATH /opt/conda/bin:$PATH
ENV PATH /opt/conda/envs/idp/bin:$PATH

# Add conda environment files (.yml)
# COPY ["./conda_environments/", "."]

USER root
# ENV CUDA_ROOT /usr/local/cuda/bin
# Get installation file
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh -O ~/anaconda.sh

# Install anaconda at /opt/conda
RUN /bin/bash ~/anaconda.sh -b -p "/opt/conda"

# Remove installation file
RUN rm ~/anaconda.sh

# Make conda command available to all users
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
# Activate conda environment with interactive bash session
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
