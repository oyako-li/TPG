# A Dockerfile that sets up a full Gym install with test dependencies
ARG PYTHON_VERSION
FROM python:3.6

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    unzip \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    patchelf \
    ffmpeg cmake \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Download mujoco
    && mkdir /root/.mujoco \
    && cd /root/.mujoco \
    && wget -qO- 'https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz' | tar -xzvf -

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin"

COPY . /usr/local/gym/
WORKDIR /usr/local/gym/

RUN if [ "python:${PYTHON_VERSION}" = "python:3.6.15" ] ; then pip install .[box2d,classic_control,toy_text,other] pytest=="7.0.1" --no-cache-dir; else pip install .[testing] --no-cache-dir; fi
RUN pip install -r requirements.txt
RUN pip install -e .
CMD ["/bin/bash"]