# Socio: Telegram bot image.
#
# Python 3.14 (not 3.11): requirements.txt is pinned on 3.14, and numpy==2.5.3 has no build for 3.11.
# Two stages: compilers only exist in the build stage, so the running image stays small.

FROM python:3.14-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt


FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SOCIO_DATA_DIR=/app/data \
    TZ=Asia/Kolkata \
    ORT_DISABLE_TELEMETRY=1

# sqlite3 and curl are for inspecting the database and the network from a shell in the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends sqlite3 curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --no-create-home --home-dir /app --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Code stays owned by root (read-only for appuser); appuser can write only the folders below.
COPY app/ app/
COPY scripts/ scripts/

# data: database, logs, embedding model (the named volume; a new volume copies this folder's owner).
# downloads: attachment and Drive cache. documents / skills: bind-mounted from the host.
RUN mkdir -p /app/data /app/downloads /app/documents /app/skills \
    && chown appuser:appuser /app/data /app/downloads /app/skills

USER appuser

CMD ["python", "-m", "app.telegram_bot"]
