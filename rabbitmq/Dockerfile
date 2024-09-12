FROM rabbitmq:3.13.1-management

HEALTHCHECK \
    --interval=5s \
    --timeout=10s \
    --retries=5 \
    CMD rabbitmq-diagnostics -q ping