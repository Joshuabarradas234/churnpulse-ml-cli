FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy only what we need to install the package
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir .

# Run from a mounted folder so outputs land on your machine
WORKDIR /work
ENTRYPOINT ["churnpulse"]
