# -------- base --------
FROM python:3.10-slim AS base
WORKDIR /app

# ---- build tooling required for Prophet's C++ extensions ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------- prod --------
FROM base AS prod
COPY . .
CMD ["streamlit", "run", "streamlit_app/user_agent.py", "--server.port=8501", "--server.enableXsrfProtection=false"]
