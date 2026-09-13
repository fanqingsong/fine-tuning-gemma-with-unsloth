FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/unsloth/unsloth:latest

WORKDIR /workspace/work

# Official image starts Jupyter by default; this project runs Python scripts.
USER root
ENTRYPOINT []
CMD ["python", "src/train.py"]
