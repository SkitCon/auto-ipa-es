# Build from MFA image for compatibility (this is the most temperamental install)
FROM mmcauliffe/montreal-forced-aligner:latest

LABEL author="Amber Converse"

# This will be our default directory for subsequent commands
WORKDIR /app

USER root
RUN apt-get update

# Install MFA
# RUN mkdir -p /mfa
# RUN mamba create -p /env -c conda-forge montreal-forced-aligner

# RUN useradd -ms /bin/bash mfauser
# RUN chown -R mfauser /mfa
# RUN chown -R mfauser /env
# RUN USER mfauser
# RUN ENV MFA_ROOT_DIR=/mfa
# RUN conda run -p /env mfa server init

# RUN echo "source activate /env && mfa server start" > ~/.bashrc
# ENV PATH /env/bin:$PATH

# # Add user for PostgreSQL
# RUN useradd -ms /bin/bash mfauser
# RUN chown -R mfauser /mfa
# RUN chown -R mfauser /env
# RUN USER mfauser
# RUN ENV MFA_ROOT_DIR=/mfa
# RUN conda run -p /env mfa server init

# RUN cd ..

# Install MFA acoustic model
RUN mfa model download acoustic spanish_mfa

# Install FST for Spanish g2p
RUN wget -L https://github.com/uiuc-sst/g2ps/raw/refs/heads/master/models/spanish_4_3_2.fst.gz

# Install Phonetisaurus and dependencies
RUN apt-get install git g++ autoconf-archive make libtool -y
RUN apt-get install python-setuptools python-dev -y
RUN mkdir g2p
WORKDIR /app/g2p
RUN wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.2.tar.gz
RUN tar -xvzf openfst-1.7.2.tar.gz
WORKDIR /app/g2p/openfst-1.7.2
RUN ./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts
RUN make -j
RUN make install
RUN echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib:/usr/local/lib/fst' >> ~/.bashrc
RUN source ~/.bashrc
WORKDIR /app/g2p
RUN git clone https://github.com/AdolfVonKleist/Phonetisaurus.git
WORKDIR /app/g2p/Phonetisaurus
RUN pip install pybindgen
RUN PYTHON=python3 ./configure --enable-python
RUN make
RUN make install
WORKDIR /app/g2p/Phonetisaurus/python
RUN cp ../.libs/Phonetisaurus.so .
RUN python3 setup.py install

WORKDIR /app

# Install requirements.txt dependencies
RUN pip install -r requirements.txt

# pandas, scikit-learn, ignite, etc.
RUN conda install -y pandas ignite -c pytorch \
    && pip install -U scikit-learn tensorboardX crc32c soundfile

# next, let's install huggingface transformers, tokenizers, and the datasets library
# we'll install a specific version of transformers 
# and the latest versions of tokenizers and datasets that are compatible with that version of transformers 
RUN pip install -U transformers==4.17.0 \
    && pip install -U tokenizers datasets
# let's include ipython as a better default REPL
# and jupyter for running notebooks
RUN conda install -y ipython jupyter ipywidgets widgetsnbextension \
    && jupyter nbextension enable --py widgetsnbextension
# let's define a default command for this image.
# We'll just print the version for our PyTorch installation
CMD ["python", "-c" "\"import torch;print(torch.__version__)\""]

# copy executables to path
COPY . ./
RUN chmod u+x  scripts/* \
    && mv scripts/* /usr/local/bin/ \
    && rmdir scripts