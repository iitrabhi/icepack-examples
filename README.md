# Installation of Icepack on Docker for MAC/Windows

This repository includes a Docker script containing all the necessary dependencies for conducting simulations with [IcePack](https://github.com/icepack/icepack/tree/master). To begin, download the zip file by clicking the green `Code` button above, and then follow these instructions. If you are still getting familiar with working in a terminal, I recommend watching this [video](https://www.youtube.com/watch?v=5XgBd6rjuDQ) first.

## Getting Started
These instructions will guide you in setting up and running the project on your local machine for development and testing purposes.

### Prerequisites
To follow the examples, you must have Docker installed on your system. Please note that this process is compatible with Windows 10 Education or Professional but not Windows 10 Home.

* [Docker](https://www.docker.com/products/docker-desktop)
* [Paraview](https://www.paraview.org/download/)
* [CMDER](https://cmder.net/) (Required only for Windows)

After installation, open `cmder`, navigate to Settings (Win+Alt+P) ➡ Import, and select the `cmlab.xml` file provided in the repository.

Once Docker is installed and running, open CMDER/terminal and execute the following commands:

### Installation
- To install the Icepack command-line interface, run the following commands:

```bash
cd /path/to/this/repo
cd docker
docker build --target base -t icepack .
```

- To install the Jupyter notebook interface, execute these commands:

```bash
cd /path/to/this/repo
cd docker
docker build --target notebook -t icepack_notebook .
```

Note: Replace the variable `/path/to/this/repo` with the path to the folder containing the Dockerfile. For example, if your code is in `D:\Codes\fenics-docker-master`, run: `cd D:\Codes\fenics-docker-master`

## Running
After building the Docker image, you can start the command-line interface by running the following command:

```bash
docker run -v host_system_path:/root/ -w /root/ -it icepack
```

To launch the notebook, use the following command:

```bash
docker run -p 8888:8888 -v host_system_path:/root/ -w /root/ icepack_notebook
```

Note: Replace the variable `host_system_path` with the path to your code's folder. For example, if your code is in `D:\Codes`, to start the command-line interface, run:

```bash
docker run -v D:\Codes:/root/ -w /root/ -it icepack
```

For Mac,
```shell
docker run -v ~/codes:/root/ -w /root/ -it icepack
```

## Authors
* [Abhinav Gupta](abhigupta.io)
