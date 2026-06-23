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

### 1. Docker Container

```bash
# 1. Edit docker-compose.yml and update environment variables inline
nano docker-compose.yml

# 2. Start the docker container
docker compose up -d

```

App runs at `http://localhost:8000`.

---

### 2. Systemd Service

```bash
# 1. Create dedicated system user
sudo useradd -r -m -s /bin/false wazuh-ai

# 2. Pull the repository and change ownership
cd /opt/
sudo git clone https://github.com/jedrzejboguszynski/wazuh-ai-assistant.git
chown -R wazuh-ai:wazuh-ai /opt/wazuh-ai-assistant/
cd wazuh-ai-assistant/

# 3. Configure Python virtual environment and install packages
sudo -u wazuh-ai python3 -m venv venv
sudo -u wazuh-ai venv/bin/pip install --upgrade pip
sudo -u wazuh-ai venv/bin/pip install -r requirements.txt

# 4. Edit the .env file and update inline variables
cp /opt/wazuh-ai-assistant/env.example /opt/wazuh-ai-assistant/.env
nano /opt/wazuh-ai-assistant/.env

# 4. Create service file
sudo cp /opt/wazuh-ai-assistant/systemd/wazuh-ai-agent.service /etc/systemd/system/
sudo chown root:root /etc/systemd/system/wazuh-ai-agent.service
sudo chmod 644 /etc/systemd/system/wazuh-ai-agent.service

# 5. Enable and start
sudo systemctl daemon-reload
sudo systemctl start wazuh-ai-agent
sudo systemctl enable wazuh-ai-agent

# 6. Check status
sudo systemctl status wazuh-ai-agent
sudo journalctl -fu wazuh-ai-agent
```

App runs at `http://localhost:8000`.

---

### 3. From CLI

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Load environment
set -a; source .env; set +a

# 3. Run directly
python agentic_soc.py
```

App runs at `http://localhost:8000`. Press `Ctrl+C` to stop.
