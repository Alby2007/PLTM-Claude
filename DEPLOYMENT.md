# Deployment Guide

## Production Deployment

This guide covers deploying the Procedural LTM MVP to production with 100% benchmark accuracy.

---

## Prerequisites

### System Requirements

- **OS:** Linux (Ubuntu 20.04+), macOS 10.15+, or Windows 10+ with WSL2
- **Python:** 3.11 or higher (required for Outlines compatibility)
- **RAM:** Minimum 2GB, recommended 4GB
- **Disk:** Minimum 1GB free space
- **CPU:** 2+ cores recommended

### Software Dependencies

```bash
# Python 3.11+
python3.11 --version

# pip (latest)
pip --version

# Git (for version control)
git --version
```

---

## Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd procedural-ltm-mvp
```

### Step 2: Create Virtual Environment

```bash
# Create venv with Python 3.11
python3.11 -m venv venv311

# Activate venv
source venv311/bin/activate  # Linux/macOS
# OR
venv311\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import outlines; import transformers; import torch; print('✅ All dependencies installed')"
```

### Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration (optional - works without API key)
nano .env
```

**Environment Variables:**

```bash
# Database
DATABASE_PATH=data/memory.db

# API Configuration (optional)
ANTHROPIC_API_KEY=your_key_here  # Only needed for API fallback
ENABLE_API_FALLBACK=false  # Set to true to enable

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Performance
MAX_WORKERS=4
BATCH_SIZE=10
```

### Step 5: Initialize Database

```bash
# Database is auto-created on first run
# Or manually initialize:
python -c "from src.storage.sqlite_store import SQLiteGraphStore; import asyncio; asyncio.run(SQLiteGraphStore('data/memory.db').connect())"
```

---

## Running the Application

### Development Mode

```bash
# Start with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Or use make command
make run
```

### Production Mode

```bash
# Start with production settings
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# With logging
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4 --log-config logging.conf
```

### Using Gunicorn (Recommended for Production)

```bash
# Install gunicorn
pip install gunicorn

# Start with gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p data logs

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - DATABASE_PATH=/app/data/memory.db
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Build and Run

```bash
# Build image
docker build -t procedural-ltm:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --name procedural-ltm \
  procedural-ltm:latest

# Or use docker-compose
docker-compose up -d
```

---

## Cloud Deployment

### AWS (EC2 + ECS)

#### EC2 Deployment

```bash
# 1. Launch EC2 instance (t3.medium recommended)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-venv git

# 4. Clone and setup
git clone <repository-url>
cd procedural-ltm-mvp
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt

# 5. Run with systemd
sudo cp deployment/procedural-ltm.service /etc/systemd/system/
sudo systemctl enable procedural-ltm
sudo systemctl start procedural-ltm
```

#### Systemd Service File

```ini
[Unit]
Description=Procedural LTM API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/procedural-ltm-mvp
Environment="PATH=/home/ubuntu/procedural-ltm-mvp/venv311/bin"
ExecStart=/home/ubuntu/procedural-ltm-mvp/venv311/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

### Google Cloud (Cloud Run)

```bash
# 1. Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/procedural-ltm

# 2. Deploy to Cloud Run
gcloud run deploy procedural-ltm \
  --image gcr.io/PROJECT_ID/procedural-ltm \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Heroku

```bash
# 1. Create Heroku app
heroku create procedural-ltm

# 2. Add Procfile
echo "web: uvicorn src.api.main:app --host 0.0.0.0 --port \$PORT --workers 4" > Procfile

# 3. Deploy
git push heroku main
```

---

## Monitoring and Logging

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
{"status": "healthy"}
```

### Logging Configuration

```python
# logging.conf
[loggers]
keys=root,uvicorn

[handlers]
keys=console,file

[formatters]
keys=default

[logger_root]
level=INFO
handlers=console,file

[logger_uvicorn]
level=INFO
handlers=console,file
qualname=uvicorn
propagate=0

[handler_console]
class=StreamHandler
level=INFO
formatter=default
args=(sys.stdout,)

[handler_file]
class=handlers.RotatingFileHandler
level=INFO
formatter=default
args=('logs/app.log', 'a', 10485760, 5)

[formatter_default]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### Metrics Collection

```python
# Add to src/api/main.py
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
requests_total = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Performance Tuning

### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_subject_predicate ON atoms(subject, predicate);
CREATE INDEX idx_graph ON atoms(graph);
CREATE INDEX idx_provenance ON atoms(provenance);

-- Enable WAL mode for better concurrency
PRAGMA journal_mode=WAL;

-- Increase cache size
PRAGMA cache_size=10000;
```

### Application Tuning

```python
# src/core/config.py
class Settings(BaseSettings):
    # Worker pool
    MAX_WORKERS: int = 4
    
    # Batch processing
    BATCH_SIZE: int = 10
    
    # Cache settings
    CACHE_TTL: int = 3600
    CACHE_SIZE: int = 1000
    
    # Connection pool
    DB_POOL_SIZE: int = 10
```

### Load Balancing

```nginx
# nginx.conf
upstream procedural_ltm {
    least_conn;
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://procedural_ltm;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Security

### API Authentication

```python
# Add to src/api/main.py
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/process")
async def process_memory(
    request: ProcessRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify token
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Process request
    ...
```

### HTTPS Configuration

```bash
# Using Let's Encrypt with Certbot
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

### Rate Limiting

```python
# Add to src/api/main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/process")
@limiter.limit("100/minute")
async def process_memory(request: Request, ...):
    ...
```

---

## Backup and Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_PATH="data/memory.db"

# Create backup
sqlite3 $DB_PATH ".backup '$BACKUP_DIR/memory_$DATE.db'"

# Compress
gzip $BACKUP_DIR/memory_$DATE.db

# Keep only last 7 days
find $BACKUP_DIR -name "memory_*.db.gz" -mtime +7 -delete
```

### Restore from Backup

```bash
# Decompress backup
gunzip /backups/memory_20260130.db.gz

# Stop application
sudo systemctl stop procedural-ltm

# Restore database
cp /backups/memory_20260130.db data/memory.db

# Start application
sudo systemctl start procedural-ltm
```

---

## Troubleshooting

### Common Issues

#### 1. Import Error: No module named 'outlines'

```bash
# Solution: Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Database Locked Error

```bash
# Solution: Enable WAL mode
sqlite3 data/memory.db "PRAGMA journal_mode=WAL;"
```

#### 3. Port Already in Use

```bash
# Solution: Find and kill process
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn src.api.main:app --port 8001
```

#### 4. Out of Memory

```bash
# Solution: Reduce workers or increase RAM
uvicorn src.api.main:app --workers 2

# Or add swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Scaling

### Horizontal Scaling

```yaml
# kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: procedural-ltm
spec:
  replicas: 4
  selector:
    matchLabels:
      app: procedural-ltm
  template:
    metadata:
      labels:
        app: procedural-ltm
    spec:
      containers:
      - name: api
        image: procedural-ltm:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Database Migration

```bash
# Migrate from SQLite to PostgreSQL
# 1. Export data
sqlite3 data/memory.db .dump > dump.sql

# 2. Convert to PostgreSQL format
sed 's/INTEGER PRIMARY KEY AUTOINCREMENT/SERIAL PRIMARY KEY/' dump.sql > pg_dump.sql

# 3. Import to PostgreSQL
psql -U postgres -d procedural_ltm < pg_dump.sql
```

---

## Maintenance

### Regular Tasks

```bash
# Daily: Check logs
tail -f logs/app.log

# Daily: Backup database
./scripts/backup.sh

# Weekly: Update dependencies
pip list --outdated
pip install --upgrade <package>

# Monthly: Vacuum database
sqlite3 data/memory.db "VACUUM;"

# Monthly: Analyze performance
pytest tests/benchmarks/ -v
```

### Version Updates

```bash
# 1. Backup current version
cp -r . ../procedural-ltm-mvp-backup

# 2. Pull latest changes
git pull origin main

# 3. Update dependencies
pip install -r requirements.txt --upgrade

# 4. Run tests
pytest tests/ -v

# 5. Restart application
sudo systemctl restart procedural-ltm
```

---

## Production Checklist

- [ ] Python 3.11+ installed
- [ ] Virtual environment created
- [ ] All dependencies installed
- [ ] Environment variables configured
- [ ] Database initialized
- [ ] Tests passing (97%+)
- [ ] Benchmarks passing (100%)
- [ ] Health check endpoint working
- [ ] Logging configured
- [ ] Backups automated
- [ ] Monitoring setup
- [ ] HTTPS enabled
- [ ] Rate limiting configured
- [ ] Authentication implemented
- [ ] Load balancer configured (if needed)
- [ ] Documentation reviewed

---

## Support

For issues or questions:
- Check `ARCHITECTURE.md` for system design
- Check `TEST_RESULTS.md` for test details
- Check `FINAL_RESULTS.md` for benchmark results
- Review logs in `logs/app.log`

---

**Status:** ✅ Production-ready with 100% benchmark accuracy

*Last updated: January 30, 2026*
