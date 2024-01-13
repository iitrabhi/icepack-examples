ARG TINI_VERSION=v0.18.0

FROM firedrakeproject/firedrake-vanilla:2023-11 as base
RUN sudo apt update && sudo apt install patchelf

RUN source firedrake/bin/activate && \
    pip install git+https://github.com/icepack/Trilinos.git && \
    pip install git+https://github.com/icepack/pyrol.git  && \
    pip install git+https://github.com/icepack/icepack.git
    # && \
    # git clone https://github.com/icepack/icepack.git && \
    # pip install ./icepack

WORKDIR /home/firedrake
COPY .bashrc /home/firedrake



WORKDIR /root

# ------------------------------------------------------------------

FROM base as notebook
LABEL description="DOLFIN Jupyter Notebook"
WORKDIR /root
USER root
ARG TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    pip3 install --upgrade --no-cache-dir jupyter jupyterlab


EXPOSE 8888/tcp
ENV SHELL /bin/bash

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
########################################