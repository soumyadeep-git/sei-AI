FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY bot.py ./
COPY knowledge ./knowledge

RUN useradd --create-home app
USER app

# PUBLIC_HOST is the hostname Twilio streams audio to; Render provides
# RENDER_EXTERNAL_HOSTNAME automatically.
CMD python bot.py -t twilio --host 0.0.0.0 --port "${PORT:-7860}" \
    -x "${PUBLIC_HOST:-$RENDER_EXTERNAL_HOSTNAME}"
