FROM python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app

WORKDIR /app
COPY build-requirements.lock requirements.lock pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --require-hashes -r build-requirements.lock \
    && pip install --no-cache-dir --no-build-isolation --require-hashes -r requirements.lock \
    && pip install --no-cache-dir --no-build-isolation --no-deps .

USER 10001:10001
ENTRYPOINT ["tg-insight"]
CMD ["run"]
