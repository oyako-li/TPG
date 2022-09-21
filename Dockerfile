# Python3のイメージを基にする
FROM python:3
ENV PYTHONUNBUFFERED 1

# ビルド時に/codeというディレクトリを作成する
RUN mkdir /code

# ワークディレクトリの設定
WORKDIR /code

# requirements.txtを/code/にコピーする
ADD requirements.txt /code/

RUN apt update --fix-missing
RUN apt install -y libopenmpi-dev ssh sudo nano gcc cmake git
RUN apt install -y build-essential freeglut3 freeglut3-dev libxi-dev libxmu-dev zlib1g-dev
RUN apt install -y xvfb python3-tk
RUN apt install -y swig
RUN apt install -y xrdp
# requirements.txtを基にpip installする
RUN pip install -r requirements.txt

ADD . /code/
