FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
# Optional: set UTF-8 locale
ENV LANG=C.UTF-8

# Install Chrome and deps, using keyring instead of apt-key
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    ca-certificates \
    unzip \
    curl \
    ffmpeg \
    libicu72 \
    # Common runtime libs for Chrome/Selenium
    libglib2.0-0 \
    libnss3 \
    libgdk-pixbuf-2.0-0 \
    libgtk-3-0 \
    libasound2 \
    libu2f-udev \
    libvulkan1 \
    && mkdir -p /etc/apt/keyrings \
    && wget -qO- https://dl.google.com/linux/linux_signing_key.pub \
    | gpg --dearmor -o /etc/apt/keyrings/google-linux-signing.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-linux-signing.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
    > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome WebDriver matching the installed Chrome major version
# (If you use Selenium >= 4.6, you can skip this whole block and let Selenium Manager handle drivers.)
RUN set -eux; \
    CHROME_MAJOR_VERSION="$(google-chrome --version | sed 's/Google Chrome //; s/ //g' | cut -d. -f1)"; \
    echo "Chrome major version: ${CHROME_MAJOR_VERSION}"; \
    LATEST_DRIVER="$(curl -fsSL "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_${CHROME_MAJOR_VERSION}" || true)"; \
    if [ -z "$LATEST_DRIVER" ]; then \
    LATEST_DRIVER="$(curl -fsSL "https://chromedriver.storage.googleapis.com/LATEST_RELEASE")"; \
    fi; \
    echo "Chromedriver version: ${LATEST_DRIVER}"; \
    wget -q "https://chromedriver.storage.googleapis.com/${LATEST_DRIVER}/chromedriver_linux64.zip"; \
    unzip chromedriver_linux64.zip; \
    mv chromedriver /usr/local/bin/chromedriver; \
    chmod +x /usr/local/bin/chromedriver; \
    rm -f chromedriver_linux64.zip

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Create deps directory for dynamic dependencies
RUN mkdir -p /app/deps

# Copy application code
COPY ./app ./app

# Environment
ENV PYTHONPATH=/app
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]