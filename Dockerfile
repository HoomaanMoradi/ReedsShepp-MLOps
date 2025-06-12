FROM python:3.10.17-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip --timeout=300 -i https://mirror-pypi.runflare.com/simple
RUN pip install --no-cache-dir -e . --timeout=3000 -i https://mirror-pypi.runflare.com/simple

EXPOSE 8080

CMD ["/bin/sh", "-c", "python pipeline/run.py && python web/application.py"]