FROM apache/spark-py

USER root
COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

ENV SPARK_DRIVER_MEMORY=16G
ENV SPARK_EXECUTOR_CORES=12
ENV SPARK_EXECUTOR_MEMORY=16G
ENV SPARK_WORKER_CORES=12
ENV SPARK_WORKER_MEMORY=16G

COPY . .

CMD ["/opt/spark/bin/spark-submit", "./src/train.py -t train"]