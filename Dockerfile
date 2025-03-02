FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib and poetry
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.local/bin"

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false

RUN poetry install --no-root --no-interaction --no-ansi

COPY trip_matching/ /app/trip_matching/
COPY tests/ /app/tests/

RUN mkdir -p /app/output

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND="Agg"

WORKDIR /app

ENTRYPOINT ["python", "/app/trip_matching/main.py", "--save"]
