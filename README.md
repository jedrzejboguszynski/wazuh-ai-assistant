![Wazuh AI logo](bot_logo.png)

## What does it do?
Enables SOC teams to query security events in natural language, receive expert threat analysis, and maintain continuous awareness of their security posture.

## Main Functions

**AI Security Analysis**
- Interactive chat with a Senior SOC Analyst AI persona
- Natural language queries for alerts, vulnerabilities, and system inventory

**Real-Time Data Integration**
- Connects to OpenSearch for Wazuh data ingestion
- Maintains rolling windows of security events
- Auto-refreshes every 15 minutes

**Persistent Vector Store**
- FAISS-based semantic search across security events
- Fast startup with cached data

**Multi-Index Monitoring**
- Queries alerts, monitoring, vulnerabilities, and inventory indexes
- Filters medium+ severity events (level 5+)

**Automatic Email Security Summaries**
- Prepares a cybersecurity posture analysis based on the data
- Sends automated reports at customizable time

## Tech Stack

- **Backend:** FastAPI (Python) with WebSocket support
- **AI:** LangChain + AI model of choice via LiteLLM + HuggingFace Embeddings
- **Vector DB:** FAISS
- **Data Source:** OpenSearch (Wazuh)
- **Email Client:** SendGrid


## In-depth Instructions
Detailed setup guides, advanced configuration options, and troubleshooting tips can be found on my blog: [Read the full guide here](https://yourbloglink.com).

## Deployment Guide

# 1. Docker Container

```bash
# 1. Configure environment
cp .env .env.backup
# Edit docker-compose.yml and update environment variables inline

# 2. Build and start
docker compose build
docker compose up -d

```

App runs at `http://localhost:8000`.

---

## 2. systemd Service

```bash
# 1. Create dedicated system user
sudo useradd -r -s /usr/sbin/nologin -m -d /home/wazuh-ai wazuh-ai

# 2. Create virtual environment
python3 -m venv /opt/wazuh-ai-agent/venv
source /opt/wazuh-ai-agent/venv/bin/activate
pip install -r /opt/wazuh-ai-agent/requirements.txt

# 3. Copy files and set ownership
sudo cp agentic_soc.py .env bot_logo.png /opt/wazuh-ai-agent/
sudo mkdir -p /opt/wazuh-ai-agent/data /opt/wazuh-ai-agent/.cache/huggingface
sudo chown -R wazuh-ai:wazuh-ai /opt/wazuh-ai-agent

# 4. Edit the .env file and update inline variables
nano .env

# 4. Create service file
sudo tee /etc/systemd/system/wazuh-ai-agent.service << 'EOF'
[Unit]
Description=Wazuh AI Agent - Security Operations Center Assistant
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=wazuh-ai
Group=wazuh-ai
WorkingDirectory=/opt/wazuh-ai-agent
Environment="PATH=/opt/wazuh-ai-agent/venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/opt/wazuh-ai-agent/.env

# Start command
ExecStart=/opt/wazuh-ai-agent/venv/bin/python3 /opt/wazuh-ai-agent/agentic_soc.py

# Restart policy
Restart=always
RestartSec=10
StartLimitInterval=0

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=wazuh-ai-agent

[Install]
WantedBy=multi-user.target
EOF

# 5. Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now wazuh-ai-agent

# 6. Check status
sudo systemctl status wazuh-ai-agent
sudo journalctl -u wazuh-ai-agent -f
```

App runs at `http://localhost:8000`.

---

## 3. From CLI

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Load environment
set -a; source .env; set +a

# 3a. Run directly
python agentic_soc.py

# 3b. Run as daemon
python agentic_soc.py --daemon
```

App runs at `http://localhost:8000`. Press `Ctrl+C` to stop.