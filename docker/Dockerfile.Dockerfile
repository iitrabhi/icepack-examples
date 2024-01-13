FROM firedrakeproject/firedrake-vanilla:2023-11
RUN sudo apt update && sudo apt install patchelf
RUN source firedrake/bin/activate && \
    pip install git+https://github.com/icepack/Trilinos.git && \
    pip install git+https://github.com/icepack/pyrol.git && \
    git clone https://github.com/icepack/icepack.git && \
    pip install --editable ./icepack