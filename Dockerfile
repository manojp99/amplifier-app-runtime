FROM python:3.12-slim

# Install git (needed for pip installing git+ dependencies)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Copy runtime source
WORKDIR /app
COPY . /app

# Install the runtime and all dependencies
RUN uv pip install --system -e ".[dev]"

# Expose the HTTP port
EXPOSE 4096

# No ~/.amplifier/cache exists - completely clean environment
# Run with --http flag for SDK access
CMD ["python", "-m", "amplifier_app_runtime.cli", "--http", "--port", "4096", "--host", "0.0.0.0"]
