FROM continuumio/miniconda3
RUN mkdir -p /home/repos
RUN mkdir -p /home/envs
COPY /lfs/local/0/jdunnmon/repos/snorkel /home/repos 
COPY /lfs/local/0/jdunnmon/repos/chtap /home/repos

RUN conda env create -f /home/repos/chtap/chtap.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /home/repos/chtap/chatp.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /home/repos/chtap/chtap.yml | cut -d' ' -f2)/bin:$PATH
