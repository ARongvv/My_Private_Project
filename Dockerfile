FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ADD . /app

# 设置工作目录
WORKDIR /app

# 设置pip国内源，尝试不同的源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# 逐个安装依赖包，以便找出冲突的包
COPY requirements.txt .
RUN cat requirements.txt | xargs -n 1 pip install

# 定义容器启动时默认执行的命令
CMD ["sh", "/app/run.sh"]