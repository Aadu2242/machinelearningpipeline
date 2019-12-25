FROM python:3.7.2-slim
COPY . /machinelearningpipeline

WORKDIR /machinelearningpipeline

RUN pip install -r requirements.txt

CMD echo "run any .py file for example python miners/imdb_miner.py"