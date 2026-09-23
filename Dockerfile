FROM python:3.11-slim

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY openmoe ./openmoe
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .

CMD ["python", "scripts/train_baseline.py", "--config", "configs/smoke_cpu.yaml", "--steps", "10"]
