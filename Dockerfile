FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir .

WORKDIR /work
ENTRYPOINT ["python", "-m", "churnpulse.cli"]

