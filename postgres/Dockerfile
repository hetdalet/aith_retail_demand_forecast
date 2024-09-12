FROM postgres:14.3-alpine

HEALTHCHECK \
    --interval=5s \
    --timeout=10s \
    --retries=5 \
    CMD pg_isready -p ${DB_PORT}

