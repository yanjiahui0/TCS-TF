FROM python:3.11-slim
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -e .
CMD ["python", "scripts/run_s1_aliasing.py", "--n", "5000", "--out", "outputs/s1.json"]
