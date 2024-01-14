ARG TINI_VERSION=v0.18.0

FROM firedrakeproject/firedrake-vanilla:2023-11 as base
RUN sudo apt update && sudo apt install patchelf

RUN source firedrake/bin/activate && \
    pip install git+https://github.com/icepack/Trilinos.git && \
    pip install git+https://github.com/icepack/pyrol.git  && \
    pip install git+https://github.com/icepack/icepack.git && \
    pip install --upgrade --no-cache-dir jupyter jupyterlab

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
    python -m ipykernel install --user --name=firedrake --display-name='Firedrake' && \
    chmod +x /tini"

EXPOSE 8888/tcp
ENV SHELL /bin/bash

# Copy the start-notebook script into the container
COPY start-notebook.sh /start-notebook.sh
RUN chmod +x /start-notebook.sh

# Use the script as the entry point
ENTRYPOINT ["/start-notebook.sh"]


# ENTRYPOINT ["/bin/bash", "-c", "source /home/firedrake/firedrake/bin/activate && jupyter lab --ip 0.0.0.0 --no-browser --allow-root"]


# ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]