FROM ubuntu:14.04

# Installing GPG
RUN apt-get update && apt-get -y install gnupg

# Adding wget and bzip2
RUN apt-get install -y wget bzip2

# Add sudo
RUN apt-get -y install sudo

# Add the PostgreSQL PGP key to verify their Debian packages.
# It should be the same key as https://www.postgresql.org/media/keys/ACCC4CF8.asc
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8

# Add PostgreSQL's repository. It contains the most recent stable release
#     of PostgreSQL, ``9.3``.
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ precise-pgdg main" > /etc/apt/sources.list.d/pgdg.list

# Update the Ubuntu and PostgreSQL repository indexes and install ``python-software-properties``,
# ``software-properties-common`` and PostgreSQL 9.3
# There are some warnings (in red) that show up during the build. You can hide
# them by prefixing each apt-get statement with DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y -q install python-software-properties software-properties-common \
    && apt-get -y -q install postgresql-9.3 postgresql-client-9.3 postgresql-contrib-9.3

# Add user ubuntu with no password, add to sudo group
#RUN adduser --disabled-password --gecos '' postgres
RUN adduser postgres sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER postgres
WORKDIR /home/postgres/
RUN sudo chmod a+rwx /home/postgres/

RUN /etc/init.d/postgresql start \
    && psql --command "CREATE USER docker WITH SUPERUSER PASSWORD 'docker';" \
    && createdb docker docker

# Adjust PostgreSQL configuration so that remote connections to the
# database are possible.
RUN echo "host all  all    0.0.0.0/0  md5" >> /etc/postgresql/9.3/main/pg_hba.conf

# And add ``listen_addresses`` to ``/etc/postgresql/9.3/main/postgresql.conf``
RUN echo "listen_addresses='*'" >> /etc/postgresql/9.3/main/postgresql.conf

# Expose the PostgreSQL port
EXPOSE 5432

RUN mkdir -p /var/run/postgresql && chown -R postgres /var/run/postgresql

# Add VOLUMEs to allow backup of config, logs and databases
VOLUME  ["/etc/postgresql", "/var/log/postgresql", "/var/lib/postgresql"]

# Set the default command to run when starting the container
CMD ["/usr/lib/postgresql/9.3/bin/postgres", "-D", "/var/lib/postgresql/9.3/main", "-c", "config_file=/etc/postgresql/9.3/main/postgresql.conf"]

USER root

# Installing anaconda

# Add user ubunu with no password, add to sudo group
#RUN adduser --disabled-password --gecos '' ubuntu
#RUN adduser ubuntu sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER ubuntu
#WORKDIR /home/ubuntu/
#RUN chmod a+rwx /home/ubuntu/
#RUN echo `pwd`

# Anaconda installing
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

# Set path to conda
#ENV PATH /root/anaconda3/bin:$PATH
ENV PATH /root/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN conda update conda
RUN conda update anaconda
RUN conda update --all

# Configuring access to Jupyter
RUN mkdir /root/notebooks
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py

# Jupyter listens port: 8888
EXPOSE 8888

# Creating directories and copying code repos
RUN sudo mkdir -p /home/repos
COPY extractors /home/repos/extractors

# Installing git
RUN apt-get install -y git

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

# Activating environment
RUN /bin/bash -c "source ~/.bashrc"
RUN /bin/bash -c "source activate chtap"

# Obtaining English language model fro spacy
RUN pip install --upgrade pip
RUN conda install spacy=1.9.0
RUN python -m spacy download en

# Installing snorkel
COPY snorkel /home/repos/snorkel
WORKDIR /home/repos/snorkel
#RUN export LANG=C.UTF-8
RUN pip install --upgrade pip && LC_ALL="en_US.UTF-8" pip install .
