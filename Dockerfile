FROM python:latest
LABEL authors="Jodai Caignie"

ENTRYPOINT ["top", "-b"]