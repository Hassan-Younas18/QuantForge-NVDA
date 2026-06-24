# Build context is the project root (see docker-compose.yml: build.context: .)
# so this can COPY both the core pipeline (src/, main.py) and the API (backend/).
FROM python:3.11-slim

WORKDIR /app

# System deps for scientific Python wheels (numpy/scipy/pyarrow) build smoothly
# on slim images without extra packages on recent manylinux wheels, so none
# are needed here beyond pip itself.

COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r backend-requirements.txt

COPY main.py ./main.py
COPY src/ ./src/
COPY backend/ ./backend/
COPY tests/ ./tests/

# outputs/ is volume-mounted at runtime (see docker-compose.yml) so trained
# models/cache survive container restarts; create it here for first boot.
RUN mkdir -p outputs/models/scalers outputs/results outputs/plots outputs/data_cache

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
