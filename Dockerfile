FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pdm

COPY pyproject.toml pdm.lock ./

RUN pdm install --prod --no-lock --no-editable

COPY . .

RUN mkdir -p /app/outputs

# Matplotlib backend
ENV MPLBACKEND=Agg

CMD ["pdm", "run", "python", "__main__.py"]