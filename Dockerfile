FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for matplotlib and poetry
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.local/bin"

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false

RUN poetry install --no-dev --no-interaction --no-ansi

COPY trip_matching/ /app/trip_matching/
COPY main.py /app/

RUN mkdir -p /app/output

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND="Agg"

WORKDIR /app/output

ENTRYPOINT ["python", "/app/main.py", "--save"]
