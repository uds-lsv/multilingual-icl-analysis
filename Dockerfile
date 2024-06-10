# Base image must at least have pytorch and CUDA installed.
# We are using NVIDIA NGC's PyTorch image here, see: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch for latest version
# See https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2021 for installed python, pytorch, etc. versions

FROM nvcr.io/nvidia/pytorch:22.12-py3

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

# Install additional programs
RUN apt update && \
    apt install -y build-essential \
    htop \
    gnupg \
    curl \
    ca-certificates \
    vim \
    tmux && \
    rm -rf /var/lib/apt/lists

# Update pip
RUN SHA=ToUcHMe which python3
RUN python3 -m pip install --upgrade pip

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install dependencies
RUN python3 -m pip install autopep8
RUN python3 -m pip install attrdict
RUN python3 -m pip install h5py
RUN python3 -m pip install jsonlines
RUN python3 -m pip install rich
RUN python3 -m pip install wandb
RUN python3 -m pip install plotly
RUN python3 -m pip install pytablewriter

# Install additional dependencies
RUN python3 -m pip install transformers>=4.28.1
RUN python3 -m pip install datasets>=2.7.1
RUN python3 -m pip install evaluate>=0.3.0
RUN python3 -m pip install faiss_gpu>=1.7.2
RUN python3 -m pip install nltk>=3.8
RUN python3 -m pip install openai>=0.27.1
RUN python3 -m pip install rank_bm25>=0.2.2
RUN python3 -m pip install requests>=2.28.1
RUN python3 -m pip install sentence_transformers>=2.2.2
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install accelerate
RUN python3 -m pip install bitsandbytes
RUN python3 -m pip install tqdm
RUN python3 -m pip install tiktoken
RUN python3 -m pip install aiolimiter
RUN python3 -m pip install sacrebleu

# Specify a new user
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

RUN mkdir -p /home/$USER_NAME
RUN useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME
RUN mkdir /home/$USER_NAME/.local

RUN chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

CMD ["/bin/bash"]

