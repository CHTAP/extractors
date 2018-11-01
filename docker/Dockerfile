FROM continuumio/miniconda3

# Creating directories and copying code repos
RUN mkdir -p /home/repos
COPY snorkel /home/repos/snorkel
COPY extractors /home/repos/extractors

# Installing GCC
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fonts-dejavu \
    gfortran \
    gcc && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Creating conda environment
RUN conda env create -f /home/repos/extractors/chtap.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /home/repos/extractors/chtap.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /home/repos/extractors/chtap.yml | cut -d' ' -f2)/bin:$PATH

# Obtaining English language model fro spacy
RUN pip install --upgrade pip
RUN conda install spacy=1.9.0
RUN python -m spacy download en

# Installing snorkel
RUN pip install -e /home/repos/snorkel
