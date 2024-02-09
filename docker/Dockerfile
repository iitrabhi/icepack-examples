ARG TINI_VERSION=v0.18.0

FROM firedrakeproject/firedrake-vanilla:2023-11 as base
RUN sudo apt update && sudo apt install patchelf

RUN source firedrake/bin/activate && \
    pip install git+https://github.com/icepack/Trilinos.git && \
    pip install git+https://github.com/icepack/pyrol.git  && \
    pip install git+https://github.com/icepack/icepack.git && \
    pip install --upgrade --no-cache-dir jupyter jupyterlab && \
    pip install h5netcdf && \
    pip install netCDF4 && \
    pip install black[jupyter]

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


# Install Jupyter
RUN /bin/bash -c "source /home/firedrake/firedrake/bin/activate && \
    pip install --upgrade jupyter_client jupyter_core jupyter_server && \
    chmod +x /tini"

EXPOSE 8888/tcp
ENV SHELL /bin/bash

# Copy the start-notebook script into the container
COPY start-notebook.sh /start-notebook.sh
RUN chmod +x /start-notebook.sh

# Use the script as the entry point
ENTRYPOINT ["/start-notebook.sh"]
