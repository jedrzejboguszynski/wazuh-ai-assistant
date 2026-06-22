import json
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.schema.messages import SystemMessage, HumanMessage
import uvicorn
import argparse
import sys
from fastapi import Depends, status, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import requests
from requests.auth import HTTPBasicAuth
import threading
import time
import asyncio
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
import schedule
from datetime import datetime as dt

app = FastAPI()
security = HTTPBasic()

class Prompt(BaseModel):
    question: str

# ===== Globals for caching =====
qa_chain = None
context = None
vectorstore = None
last_refresh_time = None
oldest_record_time = None  # Track oldest record in vectorstore

# Performance optimizations applied:
# - Chunk size: 3000 (fewer documents to embed)
# - Alert filtering: Level 5+ only (medium severity and above)
# - Field exclusion: Remove unnecessary fields
# - Parallel processing: Uses all CPU cores
# - Batch embedding: 32 documents at a time

# Auto-refresh configuration (can be overridden by environment variables)
auto_refresh_enabled = os.getenv('AUTO_REFRESH_ENABLED', 'true').lower() == 'true'
auto_refresh_interval = int(os.getenv('AUTO_REFRESH_INTERVAL', '15')) * 60  # Convert minutes to seconds
refresh_thread = None  # Background thread for auto-refresh
stop_refresh = False  # Signal to stop the refresh thread

# Vector store persistence - saves in local directory where script runs
script_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_path = os.path.join(script_dir, "data", "vectorstore")  # Path to save/load FAISS index
metadata_path = os.path.join(script_dir, "data", "vectorstore_metadata.json")  # Metadata about the vectorstore
records_cache_path = os.path.join(script_dir, "data", "records_cache.pkl")  # Cache of actual records with timestamps

# Web UI Authentication (configured via environment variables)
username = os.getenv('WEB_USERNAME', '')
password = os.getenv('WEB_PASSWORD', '')

# Add session middleware for login
# Generate a random secret key for sessions (should be persistent in production)
import hashlib
SESSION_SECRET = hashlib.sha256(f"{username}{password}wazuh-ai-agent".encode()).hexdigest()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# OpenSearch Configuration (configured via environment variables)
opensearch_url = os.getenv('OPENSEARCH_URL', '')
opensearch_username = os.getenv('OPENSEARCH_USER', '')
opensearch_password = os.getenv('OPENSEARCH_PASSWORD', '')
# Multiple index patterns - you can query multiple Wazuh indexes at once
# Common Wazuh indexes:
# - wazuh-alerts-*: Security alerts
# - wazuh-monitoring-*: Agent status and monitoring
# - wazuh-states-vulnerabilities-*: Vulnerability data
# - wazuh-states-inventory-*: System inventory (packages, processes, etc.)
opensearch_indexes_str = os.getenv('OPENSEARCH_INDEXES', '')
opensearch_indexes = [idx.strip() for idx in opensearch_indexes_str.split(',')] if opensearch_indexes_str else []
opensearch_verify_ssl = os.getenv('OPENSEARCH_VERIFY_SSL', 'false').lower() == 'true'

# LiteLLM Configuration (OpenAI-compatible proxy) - configured via environment variables
litellm_api_base = os.getenv('LITELLM_API_BASE', '')
litellm_model = os.getenv('LITELLM_MODEL', '')
litellm_api_key = os.getenv('LITELLM_API_KEY', '')

# Days range (can be overridden by environment variable)
# Changed to 0.25 (6 hours) for optimal context window usage
# With 6 hours, ~87K records fit comfortably in Claude's context (vs 350K for 24h)
# This enables the AI to access ~100% of data instead of ~10-15%
days_range = float(os.getenv('DAYS_RANGE', '0.25'))  # Default to 6 hours (0.25 days)

# Email Reports Configuration (can be overridden by environment variables)
sendgrid_api_key = os.getenv('SENDGRID_API_KEY', '')
sendgrid_from_email = os.getenv('SENDGRID_FROM_EMAIL', '')
sendgrid_to_emails = os.getenv('SENDGRID_TO_EMAILS', '').split(',') if os.getenv('SENDGRID_TO_EMAILS') else []
daily_report_time = os.getenv('DAILY_REPORT_TIME', '07:00')  # UTC time in HH:MM format

# Smart auto-detection: Enable daily reports if SendGrid is fully configured
# User can explicitly disable by setting DAILY_REPORT_ENABLED=false
daily_report_enabled_env = os.getenv('DAILY_REPORT_ENABLED', '').lower()

if daily_report_enabled_env == 'false':
    # User explicitly disabled it
    daily_report_enabled = False
elif daily_report_enabled_env == 'true':
    # User explicitly enabled it
    daily_report_enabled = True
else:
    # Auto-detect: enable if all SendGrid settings are configured
    daily_report_enabled = bool(sendgrid_api_key and sendgrid_from_email and sendgrid_to_emails and len([e for e in sendgrid_to_emails if e.strip()]) > 0)

report_scheduler_thread = None
stop_report_scheduler = False

# Configuration validation
def validate_required_config():
    """Validate that all required configuration is present"""
    missing = []
    
    # Web UI Authentication
    if not username:
        missing.append("WEB_USERNAME")
    if not password:
        missing.append("WEB_PASSWORD")
    
    # OpenSearch
    if not opensearch_url:
        missing.append("OPENSEARCH_URL")
    if not opensearch_username:
        missing.append("OPENSEARCH_USER")
    if not opensearch_password:
        missing.append("OPENSEARCH_PASSWORD")
    if not opensearch_indexes:
        missing.append("OPENSEARCH_INDEXES")
    
    # LiteLLM
    if not litellm_api_base:
        missing.append("LITELLM_API_BASE")
    if not litellm_model:
        missing.append("LITELLM_MODEL")
    if not litellm_api_key:
        missing.append("LITELLM_API_KEY")
    
    if missing:
        print("\n" + "="*60)
        print("❌ CONFIGURATION ERROR")
        print("="*60)
        print("\nThe following required environment variables are missing:\n")
        for var in missing:
            print(f"  ❌ {var}")
        print("\nPlease configure these in:")
        print("  - docker-compose.yml (environment section)")
        print("  - .env file")
        print("  - Environment variables")
        print("\nSee .env.example for reference.")
        print("="*60 + "\n")
        return False
    
    print("✅ Configuration validation passed")
    return True

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    username_match = secrets.compare_digest(credentials.username, username)
    password_match = secrets.compare_digest(credentials.password, password)
    if not (username_match and password_match):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def run_daemon():
    import daemon
    # Save daemon log in local directory
    log_file_path = os.path.join(script_dir, "threat_hunter.log")
    with daemon.DaemonContext(
        stdout=open(log_file_path, 'a+'),
        stderr=open(log_file_path, 'a+')
    ):
        uvicorn.run(app, host="0.0.0.0", port=8000)

def test_opensearch_connection():
    """Test connection to OpenSearch"""
    try:
        health_url = f"{opensearch_url}/_cluster/health"
        response = requests.get(
            health_url,
            auth=HTTPBasicAuth(opensearch_username, opensearch_password),
            verify=opensearch_verify_ssl,
            timeout=10
        )
        response.raise_for_status()
        health = response.json()
        print(f"✅ Successfully connected to OpenSearch (status: {health.get('status', 'unknown')})")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to OpenSearch: {e}")
        return False

def load_logs_from_opensearch(past_days=1):
    """Load data from multiple OpenSearch indexes using scroll API for large datasets"""
    all_logs = []
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=past_days)
    
    # Format timestamps for OpenSearch (ISO 8601)
    start_timestamp = start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_timestamp = end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    print(f"📥 Fetching data from OpenSearch: {start_timestamp} to {end_timestamp}")
    print(f"📑 Querying indexes: {', '.join(opensearch_indexes)}")
    
    # Join multiple indexes with comma for multi-index search
    index_pattern = ",".join(opensearch_indexes)
    
    # OpenSearch query - using match_all with time range filter
    # Exclude large fields we don't need for analysis to speed up transfer
    # Filter low-severity alerts (level 0-4) to focus on important events
    query = {
        "_source": {
            "excludes": [
                "*.original",  # Exclude original unparsed fields
                "*.raw",       # Exclude raw duplicates
                "data.win.eventdata.scriptBlockText",  # Exclude large PowerShell blocks
                "data.vulnerability.package.architecture"  # Exclude redundant fields
            ]
        },
        "query": {
            "bool": {
                "must": [
                    {"match_all": {}}
                ],
                "filter": [
                    {
                        "range": {
                            "timestamp": {
                                "gte": start_timestamp,
                                "lte": end_timestamp
                            }
                        }
                    }
                ],
                "should": [
                    # Keep all non-alert data (monitoring, vulnerabilities, etc.)
                    {"bool": {"must_not": {"exists": {"field": "rule.level"}}}},
                    # For alerts, only keep level 5+ (medium severity and above)
                    {"range": {"rule.level": {"gte": 5}}}
                ],
                "minimum_should_match": 1
            }
        },
        "sort": [
            {"timestamp": {"order": "desc"}}
        ],
        "size": 2000  # Increased from 1000 - fewer round trips
    }
    
    try:
        # Initial search request with scroll
        search_url = f"{opensearch_url}/{index_pattern}/_search?scroll=5m"
        
        response = requests.post(
            search_url,
            auth=HTTPBasicAuth(opensearch_username, opensearch_password),
            headers={'Content-Type': 'application/json'},
            json=query,
            verify=opensearch_verify_ssl,
            timeout=60  # Increased from 30 to 60 seconds
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract scroll_id and hits
        scroll_id = data.get('_scroll_id')
        hits = data.get('hits', {}).get('hits', [])
        total_hits = data.get('hits', {}).get('total', {})
        
        # Handle both OpenSearch 1.x and 2.x total format
        if isinstance(total_hits, dict):
            total_count = total_hits.get('value', 0)
        else:
            total_count = total_hits
        
        print(f"📊 Found {total_count} records across all indexes")
        
        # Process initial batch - include index name and source data
        for hit in hits:
            record = hit['_source'].copy()
            record['_index'] = hit['_index']  # Tag with source index for context
            all_logs.append(record)
        
        print(f"📥 Fetched {len(all_logs)}/{total_count} records...")
        
        # Continue scrolling through results
        import time
        start_time = time.time()
        
        while len(hits) > 0 and scroll_id:
            scroll_url = f"{opensearch_url}/_search/scroll"
            scroll_body = {
                "scroll": "5m",
                "scroll_id": scroll_id
            }
            
            response = requests.post(
                scroll_url,
                auth=HTTPBasicAuth(opensearch_username, opensearch_password),
                headers={'Content-Type': 'application/json'},
                json=scroll_body,
                verify=opensearch_verify_ssl,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            scroll_id = data.get('_scroll_id')
            hits = data.get('hits', {}).get('hits', [])
            
            for hit in hits:
                record = hit['_source'].copy()
                record['_index'] = hit['_index']
                all_logs.append(record)
            
            if len(hits) > 0:
                elapsed = time.time() - start_time
                rate = len(all_logs) / elapsed if elapsed > 0 else 0
                eta = (total_count - len(all_logs)) / rate if rate > 0 else 0
                print(f"📥 Fetched {len(all_logs)}/{total_count} records ({rate:.0f} rec/s, ETA: {eta:.0f}s)")
        
        # Clean up scroll context
        if scroll_id:
            try:
                delete_scroll_url = f"{opensearch_url}/_search/scroll"
                requests.delete(
                    delete_scroll_url,
                    auth=HTTPBasicAuth(opensearch_username, opensearch_password),
                    headers={'Content-Type': 'application/json'},
                    json={"scroll_id": [scroll_id]},
                    verify=opensearch_verify_ssl,
                    timeout=10
                )
            except Exception:
                pass  # Ignore cleanup errors
        
        print(f"✅ Successfully loaded {len(all_logs)} records from OpenSearch")
        
        # Print breakdown by index
        index_counts = {}
        for log in all_logs:
            idx = log.get('_index', 'unknown')
            index_counts[idx] = index_counts.get(idx, 0) + 1
        
        for idx, count in index_counts.items():
            print(f"  📋 {idx}: {count} records")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data from OpenSearch: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    return all_logs

def create_vectorstore(logs, embedding_model):
    """Create FAISS vectorstore from Wazuh data (alerts, vulnerabilities, monitoring, etc.)"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing
    
    # Larger chunks = fewer documents to process (increased to 3000 for better performance)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    
    print(f"📦 Processing {len(logs)} records...")
    
    def process_log_batch(log_batch):
        """Process a batch of logs in parallel"""
        batch_documents = []
        for log in log_batch:
            if isinstance(log, dict):
                index_name = log.get('_index', 'unknown')
                log_copy = {k: v for k, v in log.items() if k != '_index'}
                # Use compact JSON (no indent) - reduces text size by ~40%
                log_text = f"[Index: {index_name}] {json.dumps(log_copy, separators=(',', ':'))}"
            else:
                log_text = str(log)
            
            splits = text_splitter.split_text(log_text)
            for chunk in splits:
                batch_documents.append(Document(page_content=chunk))
        return batch_documents
    
    # Split logs into batches for parallel processing
    cpu_count = min(multiprocessing.cpu_count(), 4)  # Limit to 4 to avoid issues
    batch_size = max(1000, len(logs) // (cpu_count * 4))
    batches = [logs[i:i + batch_size] for i in range(0, len(logs), batch_size)]
    
    print(f"⚡ Using {cpu_count} CPU cores, processing {len(batches)} batches...")
    
    documents = []
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        futures = [executor.submit(process_log_batch, batch) for batch in batches]
        
        completed = 0
        for future in as_completed(futures):
            batch_docs = future.result()
            documents.extend(batch_docs)
            completed += 1
            if completed % 5 == 0 or completed == len(batches):
                print(f"   Progress: {completed}/{len(batches)} batches ({len(documents)} chunks so far)")
    
    if not documents:
        print("⚠️ No documents to create vectorstore")
        return None
    
    print(f"📦 Creating FAISS index with {len(documents)} document chunks...")
    
    # Use the simpler approach to avoid segfaults
    # LangChain's built-in method is more stable than manual FAISS manipulation
    try:
        print(f"⚡ Generating embeddings (this may take a while)...")
        vectorstore = FAISS.from_documents(
            documents, 
            embedding_model,
            distance_strategy="COSINE"
        )
        print(f"✅ FAISS index created successfully")
        return vectorstore
    except Exception as e:
        print(f"❌ Error creating vectorstore: {e}")
        # Fallback: try without distance strategy
        try:
            print("🔄 Retrying with default settings...")
            vectorstore = FAISS.from_documents(documents, embedding_model)
            print(f"✅ FAISS index created successfully")
            return vectorstore
        except Exception as e2:
            print(f"❌ Failed to create vectorstore: {e2}")
            return None

def save_vectorstore_to_disk(vectorstore_obj, embedding_model, records_cache):
    """Save FAISS vectorstore and records cache to disk for persistence"""
    global last_refresh_time, oldest_record_time
    
    try:
        # Create directory if it doesn't exist
        Path(vectorstore_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        vectorstore_obj.save_local(vectorstore_path)
        
        # Save records cache with timestamps
        with open(records_cache_path, 'wb') as f:
            pickle.dump(records_cache, f)
        
        # Calculate oldest and newest record times
        if records_cache:
            timestamps = []
            for record in records_cache:
                if 'timestamp' in record:
                    try:
                        ts_str = record['timestamp']
                        if 'T' in ts_str:
                            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        else:
                            ts = datetime.strptime(ts_str[:10], "%Y-%m-%d")
                        timestamps.append(ts)
                    except:
                        continue
            
            if timestamps:
                oldest_record_time = min(timestamps)
                newest_record_time = max(timestamps)
            else:
                oldest_record_time = datetime.now() - timedelta(days=1)
                newest_record_time = datetime.now()
        else:
            oldest_record_time = datetime.now() - timedelta(days=1)
            newest_record_time = datetime.now()
        
        # Save metadata with UTC timezone
        from datetime import timezone
        current_utc = datetime.now(timezone.utc)
        
        metadata = {
            "last_refresh_time": current_utc.isoformat(),
            "oldest_record_time": oldest_record_time.isoformat() if oldest_record_time.tzinfo else oldest_record_time.replace(tzinfo=timezone.utc).isoformat(),
            "newest_record_time": newest_record_time.isoformat() if newest_record_time.tzinfo else newest_record_time.replace(tzinfo=timezone.utc).isoformat(),
            "record_count": len(records_cache),
            "days_range": days_range,
            "indexes": opensearch_indexes,
            "embedding_model": "all-MiniLM-L6-v2"
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        last_refresh_time = current_utc
        print(f"💾 Vectorstore saved to disk at {vectorstore_path}")
        print(f"📊 Cached {len(records_cache)} records from {oldest_record_time.strftime('%Y-%m-%d %H:%M')} to {newest_record_time.strftime('%Y-%m-%d %H:%M')}")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to save vectorstore to disk: {e}")
        return False

def load_vectorstore_from_disk(embedding_model):
    """Load FAISS vectorstore and records cache from disk if available"""
    global last_refresh_time, oldest_record_time
    
    try:
        # Check if vectorstore exists
        if not os.path.exists(vectorstore_path) or not os.path.exists(metadata_path):
            print("📂 No saved vectorstore found on disk")
            return None, None
        
        # Check if records cache exists
        if not os.path.exists(records_cache_path):
            print("📂 No records cache found, will rebuild")
            return None, None
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load records cache
        with open(records_cache_path, 'rb') as f:
            records_cache = pickle.load(f)
        
        # Load FAISS index
        print(f"📂 Loading vectorstore from disk")
        print(f"   Last updated: {metadata.get('last_refresh_time', 'unknown')}")
        print(f"   Records: {metadata.get('record_count', 0)}")
        print(f"   Time range: {metadata.get('oldest_record_time', 'unknown')} to {metadata.get('newest_record_time', 'unknown')}")
        
        vectorstore_obj = FAISS.load_local(
            vectorstore_path, 
            embedding_model,
            allow_dangerous_deserialization=True  # Required for pickle
        )
        
        # Update global variables - ensure timezone-aware
        from datetime import timezone
        last_refresh_time = datetime.fromisoformat(metadata['last_refresh_time'])
        if last_refresh_time.tzinfo is None:
            last_refresh_time = last_refresh_time.replace(tzinfo=timezone.utc)
            
        if 'oldest_record_time' in metadata:
            oldest_record_time = datetime.fromisoformat(metadata['oldest_record_time'])
            if oldest_record_time.tzinfo is None:
                oldest_record_time = oldest_record_time.replace(tzinfo=timezone.utc)
        
        # Calculate time since last refresh in UTC
        current_utc = datetime.now(timezone.utc)
        time_diff = (current_utc - last_refresh_time).total_seconds() / 60
        
        print(f"✅ Vectorstore and cache loaded from disk successfully")
        print(f"   📅 Last refresh was: {last_refresh_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"   ⏰ Time since last refresh: {time_diff:.1f} minutes")
        return vectorstore_obj, records_cache
        
    except Exception as e:
        print(f"⚠️ Failed to load vectorstore from disk: {e}")
        return None, None

def filter_records_by_24h_window(records_cache):
    """Remove records older than the configured time window (days_range) from cache"""
    global days_range
    now = datetime.now()
    cutoff_time = now - timedelta(days=days_range)
    
    filtered_records = []
    removed_count = 0
    
    for record in records_cache:
        if 'timestamp' in record:
            try:
                ts_str = record['timestamp']
                if 'T' in ts_str:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                else:
                    ts = datetime.strptime(ts_str[:10], "%Y-%m-%d")
                
                # Keep only records within the configured time window
                if ts >= cutoff_time:
                    filtered_records.append(record)
                else:
                    removed_count += 1
            except:
                # If can't parse timestamp, keep the record
                filtered_records.append(record)
        else:
            # No timestamp, keep the record
            filtered_records.append(record)
    
    if removed_count > 0:
        hours = days_range * 24
        print(f"🗑️  Removed {removed_count} records older than {hours:.1f} hours")
    
    return filtered_records

def rebuild_vectorstore_from_cache(records_cache, embedding_model):
    """Rebuild vectorstore from filtered records cache"""
    print(f"🔄 Rebuilding vectorstore from {len(records_cache)} cached records...")
    
    if not records_cache:
        return None
    
    vectorstore_obj = create_vectorstore(records_cache, embedding_model)
    return vectorstore_obj

def sync_new_alerts_incremental(records_cache, embedding_model):
    """Fetch new alerts since last refresh and merge with existing data"""
    global last_refresh_time, vectorstore
    
    if not last_refresh_time:
        print("⚠️ No last refresh time available")
        return None, records_cache
    
    # Use UTC time to match OpenSearch timestamps
    from datetime import timezone
    current_time_utc = datetime.now(timezone.utc)
    
    # Convert last_refresh_time to UTC if it isn't already
    if last_refresh_time.tzinfo is None:
        # Assume it was stored as UTC but without timezone info
        last_refresh_utc = last_refresh_time.replace(tzinfo=timezone.utc)
    else:
        last_refresh_utc = last_refresh_time.astimezone(timezone.utc)
    
    # Fetch new data since last refresh
    print(f"🔄 Syncing new alerts since {last_refresh_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}...")
    print(f"   Current time: {current_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Format timestamps in ISO format for OpenSearch (without microseconds)
    start_timestamp = last_refresh_utc.strftime("%Y-%m-%dT%H:%M:%S")
    end_timestamp = current_time_utc.strftime("%Y-%m-%dT%H:%M:%S")
    
    print(f"   Querying: {start_timestamp} to {end_timestamp} (UTC)")
    
    new_logs = []
    index_pattern = ",".join(opensearch_indexes)
    
    # First, do a count query to see if there are ANY new records
    count_query = {
        "query": {
            "range": {
                "timestamp": {
                    "gt": start_timestamp,
                    "lte": end_timestamp
                }
            }
        }
    }
    
    try:
        count_url = f"{opensearch_url}/{index_pattern}/_count"
        count_response = requests.post(
            count_url,
            auth=HTTPBasicAuth(opensearch_username, opensearch_password),
            headers={'Content-Type': 'application/json'},
            json=count_query,
            verify=opensearch_verify_ssl,
            timeout=30
        )
        count_response.raise_for_status()
        total_in_range = count_response.json().get('count', 0)
        print(f"   Total records in time range (before filters): {total_in_range}")
    except Exception as e:
        print(f"   ⚠️  Could not get count: {e}")
        total_in_range = "unknown"
    
    query = {
        "_source": {
            "excludes": [
                "*.original",
                "*.raw",
                "data.win.eventdata.scriptBlockText",
                "data.vulnerability.package.architecture"
            ]
        },
        "query": {
            "bool": {
                "must": [{"match_all": {}}],
                "filter": [{
                    "range": {
                        "timestamp": {
                            "gt": start_timestamp,  # Use 'gt' (greater than) to avoid duplicates
                            "lte": end_timestamp
                        }
                    }
                }],
                "should": [
                    # Keep all non-alert data
                    {"bool": {"must_not": {"exists": {"field": "rule.level"}}}},
                    # For alerts, only keep level 5+
                    {"range": {"rule.level": {"gte": 5}}}
                ],
                "minimum_should_match": 1
            }
        },
        "sort": [{"timestamp": {"order": "desc"}}],
        "size": 2000
    }
    
    try:
        search_url = f"{opensearch_url}/{index_pattern}/_search?scroll=5m"
        response = requests.post(
            search_url,
            auth=HTTPBasicAuth(opensearch_username, opensearch_password),
            headers={'Content-Type': 'application/json'},
            json=query,
            verify=opensearch_verify_ssl,
            timeout=60  # Increased from 30 to 60 seconds
        )
        response.raise_for_status()
        data = response.json()
        
        scroll_id = data.get('_scroll_id')
        hits = data.get('hits', {}).get('hits', [])
        total_hits = data.get('hits', {}).get('total', {})
        
        if isinstance(total_hits, dict):
            total_count = total_hits.get('value', 0)
        else:
            total_count = total_hits
        
        print(f"📊 Found {total_count} new records")
        
        if total_count == 0:
            print("✅ No new alerts to sync")
            return vectorstore, records_cache
        
        # Limit sync to prevent timeouts (max 50k records per sync)
        max_sync_records = 50000
        if total_count > max_sync_records:
            print(f"⚠️  Large sync detected ({total_count} records), limiting to {max_sync_records} most recent")
            effective_limit = max_sync_records
        else:
            effective_limit = total_count
        
        # Collect all new records
        for hit in hits:
            record = hit['_source'].copy()
            record['_index'] = hit['_index']
            new_logs.append(record)
        
        # Continue scrolling (but stop at limit)
        while len(hits) > 0 and scroll_id and len(new_logs) < effective_limit:
            scroll_url = f"{opensearch_url}/_search/scroll"
            scroll_body = {"scroll": "5m", "scroll_id": scroll_id}
            
            response = requests.post(
                scroll_url,
                auth=HTTPBasicAuth(opensearch_username, opensearch_password),
                headers={'Content-Type': 'application/json'},
                json=scroll_body,
                verify=opensearch_verify_ssl,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            scroll_id = data.get('_scroll_id')
            hits = data.get('hits', {}).get('hits', [])
            
            for hit in hits:
                record = hit['_source'].copy()
                record['_index'] = hit['_index']
                new_logs.append(record)
        
        print(f"📥 Fetched {len(new_logs)} new records")
        
        # Merge new records into cache
        updated_cache = records_cache + new_logs
        
        # Create embeddings for new logs and merge with existing vectorstore
        if new_logs and vectorstore:
            print("📦 Creating embeddings for new data...")
            new_vectorstore_part = create_vectorstore(new_logs, embedding_model)
            
            if new_vectorstore_part:
                print(f"🔗 Merging new data into vectorstore...")
                vectorstore.merge_from(new_vectorstore_part)
                print(f"✅ Successfully synced {len(new_logs)} new records")
            
            return vectorstore, updated_cache
        elif new_logs:
            # No existing vectorstore, create new one
            print("📦 Creating new vectorstore with synced data...")
            vectorstore = create_vectorstore(updated_cache, embedding_model)
            return vectorstore, updated_cache
        
    except Exception as e:
        print(f"❌ Error during sync: {e}")
        return vectorstore, records_cache
    
    return vectorstore, records_cache

def initialize_assistant_context():
    return """You are a Senior SOC (Security Operations Center) Analyst with 10+ years of experience in threat hunting, incident response, and security analysis.

Your expertise includes:
- Advanced threat detection and analysis using Wazuh SIEM
- MITRE ATT&CK framework and threat intelligence
- Vulnerability management and risk assessment
- Incident response and forensic analysis
- Security compliance (PCI-DSS, HIPAA, GDPR, ISO 27001)

You have access to real-time Wazuh data from the last 6 hours stored in the vector database:
- Security alerts and events (wazuh-alerts-*)
- Vulnerability scan results (wazuh-states-vulnerabilities-*)
- Agent monitoring and health status (wazuh-monitoring-*)
- System inventory data (wazuh-states-inventory-*)

🔴 CRITICAL: INDEX PRIORITIZATION RULES 🔴

When answering questions about security, threats, attacks, or incidents:
1. **ALWAYS prioritize data from wazuh-alerts-* index** (marked as [Index: wazuh-alerts-*])
   - This index contains ACTUAL security events, intrusions, malware, policy violations
   - This is your PRIMARY source for threat detection and incident analysis

2. **wazuh-monitoring-* index is ONLY for agent health/status**
   - Contains keepalive signals, configuration sync status, agent connectivity
   - Does NOT contain security alerts or attack information
   - Only mention monitoring data when specifically asked about "agent status", "agent health", or "connectivity"

3. **wazuh-states-vulnerabilities-* index is for CVE/vulnerability data**
   - Use for questions about CVEs, patches, vulnerabilities, misconfigurations

4. **wazuh-states-inventory-* index is for system inventory**
   - Package lists, OS information, hardware details

**Example: When asked "Are there any attacks?"**
❌ WRONG: Report monitoring data showing "agent is active, no attacks detected in monitoring logs"
✅ CORRECT: Search wazuh-alerts-* index for actual security alerts (authentication failures, intrusions, malware, etc.)

Your communication style:
- Professional and authoritative, but clear and accessible
- Provide context and explain WHY something matters
- Offer actionable recommendations with specific steps
- Prioritize threats based on actual risk, not just severity scores
- Reference specific alerts, CVEs, agents, and timestamps when relevant
- **Always cite which index the data comes from** (e.g., "According to wazuh-alerts-* index...")
- If you don't have enough information, say so and suggest what data would help

When analyzing security events:
1. Assess the actual threat level and business impact
2. Identify attack patterns and techniques (MITRE ATT&CK)
3. Recommend immediate actions for high-priority issues
4. Suggest preventive measures and security improvements
5. Consider false positives and provide balanced analysis

Remember: Your goal is to help SOC teams make informed decisions quickly and effectively."""

def setup_chain(past_days=0.25, force_reload=False):
    """
    Setup chain with rolling time window and incremental sync
    - Loads existing data from disk
    - Removes records older than the configured time window (default: 6 hours)
    - Syncs new records since last refresh
    - Rebuilds vectorstore if needed
    """
    global qa_chain, context, days_range, vectorstore
    days_range = past_days
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    records_cache = []
    
    # Test OpenSearch connection first
    if not test_opensearch_connection():
        print("❌ Failed to connect to OpenSearch. Skipping chain setup.")
        return
    
    # Try to load from disk first (unless force_reload is True)
    if not force_reload:
        print("🔄 Loading existing data from disk...")
        vectorstore, records_cache = load_vectorstore_from_disk(embedding_model)
        
        if vectorstore and records_cache:
            print(f"✅ Loaded {len(records_cache)} cached records")
            
            # Step 1: Remove records older than the configured time window
            hours = days_range * 24
            print(f"🕐 Applying {hours:.1f}-hour rolling window...")
            original_count = len(records_cache)
            records_cache = filter_records_by_24h_window(records_cache)
            
            # If records were removed, rebuild vectorstore
            if len(records_cache) < original_count:
                print(f"🔄 Rebuilding vectorstore after removing old records...")
                vectorstore = rebuild_vectorstore_from_cache(records_cache, embedding_model)
            
            # Step 2: Sync new alerts since last refresh
            print("🔄 Checking for new alerts...")
            vectorstore, records_cache = sync_new_alerts_incremental(records_cache, embedding_model)
            
            # Step 3: Save updated data to disk
            if vectorstore and records_cache:
                save_vectorstore_to_disk(vectorstore, embedding_model, records_cache)
                print(f"✅ Rolling window maintained: {len(records_cache)} records in last {hours:.1f}h")
            
        else:
            print("📥 No cached data found, will perform initial load from OpenSearch")
            force_reload = True
    
    # If no cached data or force_reload, fetch fresh from OpenSearch
    if not vectorstore or force_reload:
        print(f"🔄 Loading fresh data from OpenSearch (past {past_days} days)...")
        
        logs = load_logs_from_opensearch(past_days)
        if not logs:
            print("❌ No data found. Skipping chain setup.")
            return

        print(f"✅ {len(logs)} records loaded from the last {past_days} day(s).")
        records_cache = logs
        
        vectorstore = create_vectorstore(logs, embedding_model)
        
        if not vectorstore:
            print("❌ Failed to create vectorstore. Skipping chain setup.")
            return
        
        # Save to disk for future use
        save_vectorstore_to_disk(vectorstore, embedding_model, records_cache)

    if not vectorstore:
        print("❌ No vectorstore available. Skipping chain initialization.")
        return
    
    # Initialize with LiteLLM proxy (OpenAI-compatible)
    print(f"🤖 Initializing LiteLLM proxy with model '{litellm_model}' at {litellm_api_base}...")
    
    # Use ChatOpenAI with custom base_url pointing to LiteLLM proxy
    # LiteLLM proxy is OpenAI-compatible, so this works perfectly
    llm = ChatOpenAI(
        model=litellm_model,
        openai_api_key=litellm_api_key,
        openai_api_base=litellm_api_base,
        temperature=0.7,
        max_tokens=2000
    )
    
    context = initialize_assistant_context()
    
    # Configure retriever to fetch more documents for better coverage
    # k=30 ensures we get diverse data from different indexes (alerts, monitoring, vulns)
    # This increases the chance of getting wazuh-alerts-* data in the context
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 30}  # Retrieve top 30 most relevant documents (was default 4)
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    print("✅ QA chain initialized successfully with Claude via LiteLLM.")
    print(f"   Retriever configured to fetch top 30 documents for comprehensive analysis")

def refresh_with_new_alerts():
    """
    Refresh with rolling 24h window - calls setup_chain which handles:
    - Removing old records (>24h)
    - Syncing new alerts
    - Rebuilding vectorstore
    """
    try:
        # Call setup_chain without force_reload - it will do incremental sync
        setup_chain(past_days=days_range, force_reload=False)
        return True
    except Exception as e:
        print(f"❌ Error during refresh: {e}")
        return False

def auto_refresh_worker():
    """
    Background worker that automatically refreshes data every N minutes
    Maintains rolling 24h window by:
    - Removing old records
    - Syncing new alerts
    - Updating vectorstore
    """
    global stop_refresh
    
    print(f"🔄 Auto-refresh enabled: Will sync every {auto_refresh_interval // 60} minutes")
    
    # Track consecutive failures for exponential backoff
    consecutive_failures = 0
    max_failures = 3
    
    while not stop_refresh:
        # Wait for the interval (check every second to allow quick shutdown)
        for _ in range(auto_refresh_interval):
            if stop_refresh:
                break
            time.sleep(1)
        
        if stop_refresh:
            break
        
        # Perform automatic refresh with retry logic
        try:
            print(f"\n⏰ Auto-refresh triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            # Retry up to 3 times with exponential backoff
            success = False
            for attempt in range(max_failures):
                try:
                    success = refresh_with_new_alerts()
                    if success:
                        consecutive_failures = 0  # Reset failure counter
                        break
                except requests.exceptions.Timeout:
                    print(f"⚠️  Attempt {attempt + 1}/{max_failures}: Timeout")
                    if attempt < max_failures - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        print(f"⏳ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                except Exception as e:
                    print(f"⚠️  Attempt {attempt + 1}/{max_failures}: {str(e)[:100]}")
                    if attempt < max_failures - 1:
                        wait_time = 2 ** attempt
                        print(f"⏳ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
            
            if success:
                print("✅ Auto-refresh completed successfully")
                print(f"⏰ Next refresh in {auto_refresh_interval // 60} minutes")
            else:
                consecutive_failures += 1
                print(f"❌ Auto-refresh failed after {max_failures} attempts")
                print(f"⚠️  Consecutive failures: {consecutive_failures}")
                
                # If too many consecutive failures, increase interval temporarily
                if consecutive_failures >= 3:
                    print(f"⚠️  Too many failures, will try again in {(auto_refresh_interval * 2) // 60} minutes")
                    time.sleep(auto_refresh_interval)  # Double the wait time
                    consecutive_failures = 0  # Reset after extended wait
                else:
                    print(f"⏰ Will retry in {auto_refresh_interval // 60} minutes")
            
            print("=" * 60 + "\n")
            
        except Exception as e:
            print(f"❌ Critical auto-refresh error: {e}")
            print(f"⏰ Will retry in {auto_refresh_interval // 60} minutes")
            import traceback
            traceback.print_exc()
    
    print("🛑 Auto-refresh worker stopped")

def start_auto_refresh():
    """Start the auto-refresh background thread"""
    global refresh_thread, stop_refresh
    
    if not auto_refresh_enabled:
        print("⏸️  Auto-refresh is disabled")
        return
    
    if refresh_thread and refresh_thread.is_alive():
        print("⚠️  Auto-refresh already running")
        return
    
    stop_refresh = False
    refresh_thread = threading.Thread(target=auto_refresh_worker, daemon=True, name="AutoRefresh")
    refresh_thread.start()
    print(f"✅ Auto-refresh started (interval: {auto_refresh_interval // 60} minutes)")

def stop_auto_refresh():
    """Stop the auto-refresh background thread"""
    global stop_refresh
    
    print("🛑 Stopping auto-refresh...")
    stop_refresh = True
    
    if refresh_thread and refresh_thread.is_alive():
        refresh_thread.join(timeout=5)
        print("✅ Auto-refresh stopped")

def get_stats(logs):
    """Generate statistics from Wazuh alerts"""
    total_logs = len(logs)
    
    # Extract timestamps from Wazuh API alerts
    dates = []
    for log in logs:
        if 'timestamp' in log and log.get('timestamp'):
            try:
                # Wazuh API returns ISO format timestamps
                timestamp_str = log['timestamp']
                # Handle both full ISO format and date-only format
                if 'T' in timestamp_str:
                    date_obj = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    date_obj = datetime.strptime(timestamp_str[:10], "%Y-%m-%d")
                dates.append(date_obj)
            except Exception:
                continue
    
    date_range = ""
    if dates:
        earliest = min(dates).strftime("%Y-%m-%d %H:%M:%S")
        latest = max(dates).strftime("%Y-%m-%d %H:%M:%S")
        date_range = f" from {earliest} to {latest}"
    
    return f"Alerts loaded: {total_logs}{date_range}"

def diagnose_vectorstore_content():
    """Diagnose what type of data is in the vectorstore"""
    global vectorstore
    
    if not vectorstore:
        return "❌ Vectorstore not initialized"
    
    try:
        # Sample some documents from the vectorstore
        # Search for different types of content
        test_queries = [
            ("Wazuh alert", "wazuh-alerts"),
            ("vulnerability CVE", "vulnerabilities"),
            ("rule level severity", "alerts with rules"),
            ("authentication failed", "auth alerts"),
            ("email message", "email data")
        ]
        
        results = []
        results.append("🔍 VECTORSTORE CONTENT DIAGNOSTIC\n")
        results.append("=" * 50)
        
        for query, description in test_queries:
            try:
                docs = vectorstore.similarity_search(query, k=3)
                if docs:
                    results.append(f"\n📋 Query: '{query}' ({description})")
                    results.append(f"   Found {len(docs)} documents")
                    # Show first 200 chars of first doc
                    sample = docs[0].page_content[:200].replace('\n', ' ')
                    results.append(f"   Sample: {sample}...")
                else:
                    results.append(f"\n❌ Query: '{query}' - No results")
            except Exception as e:
                results.append(f"\n⚠️ Query '{query}' failed: {e}")
        
        # Try to detect what indexes are present
        results.append("\n\n📊 INDEX ANALYSIS:")
        results.append("=" * 50)
        try:
            all_docs = vectorstore.similarity_search("Index:", k=20)
            index_types = {}
            for doc in all_docs:
                content = doc.page_content
                if "[Index:" in content:
                    # Extract index name
                    idx_start = content.find("[Index:") + 7
                    idx_end = content.find("]", idx_start)
                    if idx_end > idx_start:
                        idx_name = content[idx_start:idx_end].strip()
                        index_types[idx_name] = index_types.get(idx_name, 0) + 1
            
            if index_types:
                results.append("Found index types:")
                for idx, count in sorted(index_types.items(), key=lambda x: x[1], reverse=True):
                    results.append(f"   • {idx}: ~{count} chunks")
            else:
                results.append("⚠️ Could not detect index types in documents")
        except Exception as e:
            results.append(f"⚠️ Index analysis failed: {e}")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"❌ Diagnostic failed: {e}"

# ========= Report Data Extraction =========

def get_records_by_index(index_pattern):
    """Get cached records filtered by index pattern from disk"""
    try:
        if not os.path.exists(records_cache_path):
            print(f"⚠️ No records cache found at {records_cache_path}")
            return []
        
        with open(records_cache_path, 'rb') as f:
            records_cache = pickle.load(f)
        
        # Filter records by index pattern
        filtered_records = []
        for record in records_cache:
            idx = record.get('_index', '')
            # Match pattern (e.g., "wazuh-alerts-*" matches "wazuh-alerts-4.x-2024.06.18")
            pattern_prefix = index_pattern.replace('*', '')
            if idx.startswith(pattern_prefix):
                filtered_records.append(record)
        
        print(f"📊 Found {len(filtered_records)} records matching pattern '{index_pattern}'")
        return filtered_records
    
    except Exception as e:
        print(f"❌ Error loading records by index: {e}")
        return []

def get_alerts_from_cache(limit=100):
    """Get security alerts (not vulnerabilities) from wazuh-alerts-* index"""
    alerts = get_records_by_index("wazuh-alerts-")
    
    # Filter out vulnerability alerts, keep only security events
    security_alerts = []
    for alert in alerts:
        # Check if it's a vulnerability alert
        rule_groups = alert.get('rule', {}).get('groups', [])
        if isinstance(rule_groups, list) and 'vulnerability-detector' in rule_groups:
            continue  # Skip vulnerability alerts
        
        # Keep security alerts with rule.level
        if 'rule' in alert and 'level' in alert.get('rule', {}):
            security_alerts.append(alert)
    
    # Sort by severity (rule.level) descending, then by timestamp
    security_alerts.sort(
        key=lambda x: (
            -x.get('rule', {}).get('level', 0),
            x.get('timestamp', '')
        ),
        reverse=True
    )
    
    return security_alerts[:limit]

def get_vulnerabilities_from_cache(limit=100):
    """Get vulnerability data from wazuh-states-vulnerabilities-* index"""
    vulns = get_records_by_index("wazuh-states-vulnerabilities-")
    
    # Sort by CVSS score (if available) or severity
    def get_vuln_score(vuln):
        # Try to get CVSS score
        cvss = vuln.get('vulnerability', {}).get('cvss', {})
        if isinstance(cvss, dict):
            cvss3_base = cvss.get('cvss3', {}).get('base_score', 0)
            cvss2_base = cvss.get('cvss2', {}).get('base_score', 0)
            return max(cvss3_base, cvss2_base)
        return 0
    
    vulns.sort(key=get_vuln_score, reverse=True)
    
    return vulns[:limit]

def get_all_data_summary():
    """Get summary statistics from all cached records"""
    try:
        if not os.path.exists(records_cache_path):
            return {}
        
        with open(records_cache_path, 'rb') as f:
            records_cache = pickle.load(f)
        
        summary = {
            'total_records': len(records_cache),
            'by_index': {},
            'alerts_by_level': {},
            'vulnerability_count': 0,
            'affected_agents': set(),
            'time_range': {}
        }
        
        timestamps = []
        
        for record in records_cache:
            # Count by index
            idx = record.get('_index', 'unknown')
            summary['by_index'][idx] = summary['by_index'].get(idx, 0) + 1
            
            # Count alerts by level
            if 'rule' in record and 'level' in record.get('rule', {}):
                level = record['rule']['level']
                summary['alerts_by_level'][level] = summary['alerts_by_level'].get(level, 0) + 1
            
            # Count vulnerabilities
            if 'vulnerability' in record or 'wazuh-states-vulnerabilities' in idx:
                summary['vulnerability_count'] += 1
            
            # Track agents
            agent_name = record.get('agent', {}).get('name')
            if agent_name:
                summary['affected_agents'].add(agent_name)
            
            # Track timestamps
            if 'timestamp' in record:
                try:
                    ts_str = record['timestamp']
                    if 'T' in ts_str:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        timestamps.append(ts)
                except:
                    pass
        
        # Calculate time range
        if timestamps:
            summary['time_range'] = {
                'start': min(timestamps).strftime('%Y-%m-%d %H:%M:%S'),
                'end': max(timestamps).strftime('%Y-%m-%d %H:%M:%S')
            }
        
        summary['affected_agents'] = list(summary['affected_agents'])
        
        return summary
    
    except Exception as e:
        print(f"❌ Error getting data summary: {e}")
        return {}

# ========= Daily Email Reports =========

def generate_security_report():
    """Generate comprehensive security report using LLM with direct data access"""
    global qa_chain, vectorstore
    
    if not qa_chain:
        print("❌ Cannot generate report: QA chain not initialized")
        return None
    
    print("📊 Generating daily security report...")
    
    # Get data directly from cached records
    print("  📥 Loading data from cache...")
    security_alerts = get_alerts_from_cache(limit=50)  # Get top 50 alerts
    vulnerabilities = get_vulnerabilities_from_cache(limit=50)  # Get top 50 vulns
    data_summary = get_all_data_summary()
    
    print(f"  📊 Found {len(security_alerts)} security alerts")
    print(f"  📊 Found {len(vulnerabilities)} vulnerabilities")
    
    # Build consolidated data context for single-pass analysis
    print("  📋 Preparing consolidated security data...")
    
    full_data_context = "=== SECURITY DATA LOGS (WAZUH) ===\n\n"
    
    # Add Security Alerts
    full_data_context += "SECURITY ALERTS (wazuh-alerts-*):\n"
    if security_alerts:
        for i, alert in enumerate(security_alerts[:30], 1):  # Top 30 alerts
            rule = alert.get('rule', {})
            agent = alert.get('agent', {})
            full_data_context += f"\nAlert #{i}:\n"
            full_data_context += f"  Rule Level: {rule.get('level', 'N/A')}\n"
            full_data_context += f"  Rule ID: {rule.get('id', 'N/A')}\n"
            full_data_context += f"  Description: {rule.get('description', 'N/A')}\n"
            full_data_context += f"  Agent: {agent.get('name', 'N/A')} ({agent.get('ip', 'N/A')})\n"
            full_data_context += f"  Timestamp: {alert.get('timestamp', 'N/A')}\n"
            if 'data' in alert:
                data = alert['data']
                if 'srcip' in data:
                    full_data_context += f"  Source IP: {data['srcip']}\n"
                if 'srcuser' in data:
                    full_data_context += f"  Source User: {data['srcuser']}\n"
    else:
        full_data_context += "No security alerts found.\n"
    
    # Add Vulnerabilities
    full_data_context += "\n\nVULNERABILITIES (wazuh-states-vulnerabilities-*):\n"
    if vulnerabilities:
        for i, vuln in enumerate(vulnerabilities[:30], 1):  # Top 30 vulnerabilities
            vuln_data = vuln.get('vulnerability', {})
            agent = vuln.get('agent', {})
            package = vuln_data.get('package', {})
            cvss = vuln_data.get('cvss', {})
            
            full_data_context += f"\nVulnerability #{i}:\n"
            full_data_context += f"  CVE: {vuln_data.get('cve', 'N/A')}\n"
            full_data_context += f"  Severity: {vuln_data.get('severity', 'N/A')}\n"
            
            if isinstance(cvss, dict):
                cvss3 = cvss.get('cvss3', {})
                cvss2 = cvss.get('cvss2', {})
                if cvss3:
                    full_data_context += f"  CVSS3 Score: {cvss3.get('base_score', 'N/A')}\n"
                if cvss2:
                    full_data_context += f"  CVSS2 Score: {cvss2.get('base_score', 'N/A')}\n"
            
            full_data_context += f"  Agent: {agent.get('name', 'N/A')}\n"
            full_data_context += f"  Package: {package.get('name', 'N/A')} v{package.get('version', 'N/A')}\n"
            full_data_context += f"  Detection Time: {vuln.get('timestamp', 'N/A')}\n"
    else:
        full_data_context += "No vulnerabilities found.\n"
    
    # Add Summary Statistics
    full_data_context += "\n\nOVERALL SUMMARY:\n"
    full_data_context += f"Total Records: {data_summary.get('total_records', 0)}\n"
    full_data_context += f"Total Vulnerabilities: {data_summary.get('vulnerability_count', 0)}\n"
    full_data_context += f"Monitored Agents: {len(data_summary.get('affected_agents', []))}\n"
    
    if data_summary.get('alerts_by_level'):
        full_data_context += "\nAlerts by Severity:\n"
        for level in sorted(data_summary.get('alerts_by_level', {}).keys(), reverse=True):
            count = data_summary['alerts_by_level'][level]
            full_data_context += f"  Level {level}: {count} alerts\n"
    
    if 'time_range' in data_summary:
        tr = data_summary['time_range']
        full_data_context += f"\nTime Range: {tr.get('start', 'N/A')} to {tr.get('end', 'N/A')}\n"
    
    # Create the master prompt using the exact format requested
    master_prompt = f"""You are an expert Cybersecurity AI Agent. Your task is to generate a structured Cybersecurity Posture Summary based on provided security data logs (Wazuh)". 

Adhere strictly to the following three-section structure:

### Section 1: Top 5 Critical Security Alerts (Triage Priority)
- Identify and list the top 5 most important security alerts requiring immediate human investigation.
- Adopt the perspective of a Senior SOC Analyst during initial triage.
- CRITICAL CONSTRAINT: Do NOT include any vulnerability discovery alerts in this section. Focus exclusively on active threats, anomalies, or policy violations.

### Section 2: High-Level Cybersecurity Posture Executive Summary
- Provide a concise, macro-level overview of the organization's current security posture.
- Aggregated Metrics: Summarize total alert volumes and risk trends.
- CRITICAL CONSTRAINT: Do NOT list individual vulnerabilities or specific alerts here. Keep this section strictly at an executive, strategic level.

{full_data_context}"""
    
    # Generate the complete report in one pass
    print("  🤖 Generating comprehensive report...")
    
    try:
        report_response = qa_chain.invoke({
            "question": master_prompt,
            "chat_history": []
        })
        report_content = report_response.get("answer", "Unable to generate security report.")
    except Exception as e:
        print(f"  ⚠️ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        report_content = f"Error generating report: {e}"
    
    # Format the final report with header
    report_date = dt.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WAZUH DAILY SECURITY REPORT
Generated: {report_date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{report_content}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Report generated by Wazuh AI Agent
Powered by Claude via LiteLLM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    print("✅ Security report generated successfully")
    return report

def send_email_report(report_content, subject=None):
    """Send email report via SendGrid"""
    
    if not sendgrid_api_key:
        print("❌ SendGrid API key not configured")
        return False
    
    if not sendgrid_from_email:
        print("❌ SendGrid from email not configured")
        return False
    
    if not sendgrid_to_emails or len(sendgrid_to_emails) == 0:
        print("❌ No recipient emails configured")
        return False
    
    if not report_content:
        print("❌ No report content to send")
        return False
    
    # Clean up empty emails from split
    recipients = [email.strip() for email in sendgrid_to_emails if email.strip()]
    
    if len(recipients) == 0:
        print("❌ No valid recipient emails")
        return False
    
    # Default subject
    if not subject:
        subject = f"Wazuh Daily Security Report - {dt.utcnow().strftime('%Y-%m-%d')}"
    
    try:
        print(f"📧 Sending email report to {len(recipients)} recipient(s)...")
        
        # Create HTML version with better formatting
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Courier New', monospace; background-color: #1e1e1e; color: #ffffff; padding: 20px; }}
                pre {{ background-color: #2d2d2d; padding: 20px; border-radius: 5px; overflow-x: auto; }}
                .header {{ color: #3595F9; font-size: 18px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">Wazuh Daily Security Report</div>
            <pre>{report_content}</pre>
        </body>
        </html>
        """
        
        # Create message
        message = Mail(
            from_email=Email(sendgrid_from_email),
            to_emails=[To(email) for email in recipients],
            subject=subject,
            plain_text_content=Content("text/plain", report_content),
            html_content=Content("text/html", html_content)
        )
        
        # Send via SendGrid
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        
        if response.status_code in [200, 201, 202]:
            print(f"✅ Email sent successfully (status: {response.status_code})")
            print(f"   Recipients: {', '.join(recipients)}")
            return True
        else:
            print(f"⚠️  Email sent with status: {response.status_code}")
            return True  # SendGrid accepts 202 as success
            
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_and_send_daily_report():
    """Generate and send daily security report"""
    print("\n" + "="*60)
    print(f"📊 DAILY SECURITY REPORT - {dt.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*60)
    
    # Generate report
    report = generate_security_report()
    
    if report:
        # Send email
        success = send_email_report(report)
        
        if success:
            print("✅ Daily security report sent successfully!")
        else:
            print("❌ Failed to send daily security report")
    else:
        print("❌ Failed to generate daily security report")
    
    print("="*60 + "\n")
    return report is not None and success

def report_scheduler_worker():
    """Background worker for daily report scheduling"""
    global stop_report_scheduler
    
    print(f"📅 Daily report scheduler started")
    print(f"   Schedule: {daily_report_time} UTC daily")
    print(f"   Recipients: {', '.join(sendgrid_to_emails) if sendgrid_to_emails else 'None configured'}")
    
    # Schedule daily report
    schedule.every().day.at(daily_report_time).do(generate_and_send_daily_report)
    
    while not stop_report_scheduler:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
    
    print("🛑 Report scheduler stopped")

def start_report_scheduler():
    """Start the daily report scheduler"""
    global report_scheduler_thread, stop_report_scheduler
    
    if not daily_report_enabled:
        # Show why it's disabled
        daily_report_enabled_env = os.getenv('DAILY_REPORT_ENABLED', '').lower()
        if daily_report_enabled_env == 'false':
            print("⏸️  Daily reports explicitly disabled (DAILY_REPORT_ENABLED=false)")
        else:
            print("⏸️  Daily reports disabled (SendGrid not configured)")
            print("   To enable: Set SENDGRID_API_KEY, SENDGRID_FROM_EMAIL, SENDGRID_TO_EMAILS")
        return False
    
    if not sendgrid_api_key or not sendgrid_from_email or not sendgrid_to_emails:
        print("⚠️  Daily reports enabled but SendGrid not fully configured")
        print("   Required: SENDGRID_API_KEY, SENDGRID_FROM_EMAIL, SENDGRID_TO_EMAILS")
        return False
    
    if report_scheduler_thread and report_scheduler_thread.is_alive():
        print("⚠️  Report scheduler already running")
        return False
    
    stop_report_scheduler = False
    report_scheduler_thread = threading.Thread(
        target=report_scheduler_worker,
        daemon=True,
        name="ReportScheduler"
    )
    report_scheduler_thread.start()
    
    # Show if auto-enabled or explicitly enabled
    daily_report_enabled_env = os.getenv('DAILY_REPORT_ENABLED', '').lower()
    if daily_report_enabled_env == 'true':
        print(f"✅ Daily report scheduler started (sends at {daily_report_time} UTC)")
        print(f"   Explicitly enabled via DAILY_REPORT_ENABLED=true")
    else:
        print(f"✅ Daily report scheduler started (sends at {daily_report_time} UTC)")
        print(f"   Auto-enabled (SendGrid configured)")
    
    return True

def stop_report_scheduler():
    """Stop the daily report scheduler"""
    global stop_report_scheduler
    
    print("🛑 Stopping daily report scheduler...")
    stop_report_scheduler = True
    
    if report_scheduler_thread and report_scheduler_thread.is_alive():
        report_scheduler_thread.join(timeout=5)
        print("✅ Daily report scheduler stopped")
        return True
    else:
        print("⚠️  Report scheduler was not running")
        return False

# ========= WebSocket Chat =========

chat_history = []

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    global qa_chain, context, chat_history, days_range
    
    # Check authentication via cookie
    # WebSocket connections include cookies, so we can check the session
    cookie_header = websocket.headers.get('cookie', '')
    
    # Simple session check - in production, this should be more robust
    # For now, we accept the connection and let the frontend handle redirects
    await websocket.accept()

    try:
        if not context:
            await websocket.send_json({"role": "bot", "message": "⚠️ Assistant not ready yet. Please wait."})
            await websocket.close()
            return
        
        chat_history = [SystemMessage(content=context)]
        auto_status = f"🔄 Auto-refresh: ON (every {auto_refresh_interval // 60} min)" if auto_refresh_enabled else "🔄 Auto-refresh: OFF"
        
        # Daily reports status
        if daily_report_enabled and report_scheduler_thread and report_scheduler_thread.is_alive():
            reports_status = f"📧 Daily Reports: ON ({daily_report_time} UTC)"
        elif daily_report_enabled:
            reports_status = "📧 Daily Reports: Configured but not running"
        else:
            reports_status = "📧 Daily Reports: OFF"
        
        await websocket.send_json({"role": "bot", "message": f"👋 Hello! Ask me anything about Wazuh data (alerts, agent status, vulnerabilities, etc.).\n\n📊 Time range: {days_range} day(s)\n{auto_status}\n{reports_status}\n\nType /help for commands.\nPowered by Claude via LiteLLM."})

        while True:
            data = await websocket.receive_text()
            if not data.strip():
                continue
            
            # Commands handling
            if data.lower() == "/help":
                help_msg = (
                    "📋 Help Menu:\n\n"
                    "🔄 Data Management:\n"
                    "/reload - Full reload: Fetch all data from OpenSearch for current time range.\n"
                    "/refresh - Quick refresh: Fetch only NEW data since last update.\n"
                    "/set days <number> - Set number of days for data to load (1-365).\n"
                    "/stat - Show quick statistics about the loaded data.\n"
                    "/diagnose - Analyze what type of data is in the vectorstore (troubleshooting).\n\n"
                    "⏰ Auto-Refresh (Background Sync):\n"
                    "/autostatus - Check auto-refresh status.\n"
                    "/autostop - Stop automatic background refresh.\n"
                    "/autostart - Start automatic background refresh.\n\n"
                    "📧 Daily Email Reports:\n"
                    "/reportstatus - Check daily report status and configuration.\n"
                    "/reportstart - Start daily report scheduler.\n"
                    "/reportstop - Stop daily report scheduler.\n"
                    "/reportsend - Send security report immediately (force send).\n\n"
                    f"📊 Currently querying indexes: {', '.join(opensearch_indexes)}\n"
                )
                if last_refresh_time:
                    help_msg += f"⏰ Last refresh: {last_refresh_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                if auto_refresh_enabled and refresh_thread and refresh_thread.is_alive():
                    help_msg += f"🔄 Auto-refresh: ENABLED (every {auto_refresh_interval // 60} min)"
                await websocket.send_json({"role": "bot", "message": help_msg})
                continue

            if data.lower() == "/reload":
                await websocket.send_json({"role": "bot", "message": f"🔄 Full reload: Fetching all data for past {days_range} day(s) from OpenSearch..."})
                setup_chain(past_days=days_range, force_reload=True)
                if qa_chain:
                    await websocket.send_json({"role": "bot", "message": f"✅ Reload complete. Now using data from past {days_range} day(s)."})
                    chat_history = [SystemMessage(content=context)]
                else:
                    await websocket.send_json({"role": "bot", "message": "❌ Reload failed: no data found or error initializing chain."})
                continue
            
            if data.lower() == "/refresh":
                await websocket.send_json({"role": "bot", "message": "🔄 Refreshing with new alerts from last 24 hours..."})
                success = refresh_with_new_alerts()
                if success:
                    await websocket.send_json({"role": "bot", "message": "✅ Refresh complete! New alerts added to vectorstore."})
                else:
                    await websocket.send_json({"role": "bot", "message": "⚠️ Refresh failed or no new data available."})
                continue

            if data.lower().startswith("/set days"):
                try:
                    parts = data.split()
                    new_days = int(parts[-1])
                    if new_days < 1 or new_days > 365:
                        await websocket.send_json({"role": "bot", "message": "⚠️ Please specify a number between 1 and 365."})
                        continue
                    days_range = new_days
                    await websocket.send_json({"role": "bot", "message": f"✅ Date range set to {days_range} days (effective on next reload)."})
                except Exception:
                    await websocket.send_json({"role": "bot", "message": "⚠️ Invalid command format. Use: /set days <number>."})
                continue

            if data.lower() == "/stat":
                logs = load_logs_from_opensearch(days_range)
                stats = get_stats(logs)
                await websocket.send_json({"role": "bot", "message": stats})
                continue
            
            if data.lower() == "/diagnose":
                await websocket.send_json({"role": "bot", "message": "🔍 Analyzing vectorstore content..."})
                diagnosis = diagnose_vectorstore_content()
                await websocket.send_json({"role": "bot", "message": diagnosis})
                continue
            
            if data.lower() == "/autostatus":
                if auto_refresh_enabled and refresh_thread and refresh_thread.is_alive():
                    status_msg = (
                        f"🔄 Auto-refresh: ENABLED ✅\n"
                        f"⏰ Interval: {auto_refresh_interval // 60} minutes\n"
                        f"📊 Last refresh: {last_refresh_time.strftime('%Y-%m-%d %H:%M:%S') if last_refresh_time else 'Never'}\n"
                        f"🔮 Next refresh: In ~{auto_refresh_interval // 60} minutes"
                    )
                elif auto_refresh_enabled:
                    status_msg = "🔄 Auto-refresh: ENABLED but not running ⚠️"
                else:
                    status_msg = "🔄 Auto-refresh: DISABLED ❌"
                await websocket.send_json({"role": "bot", "message": status_msg})
                continue
            
            if data.lower() == "/autostop":
                global stop_refresh
                stop_refresh = True
                await websocket.send_json({"role": "bot", "message": "🛑 Auto-refresh stopped. Use /autostart to resume."})
                continue
            
            if data.lower() == "/autostart":
                start_auto_refresh()
                await websocket.send_json({"role": "bot", "message": f"✅ Auto-refresh started (interval: {auto_refresh_interval // 60} minutes)"})
                continue
            
            # Daily Report Commands
            if data.lower() == "/reportstatus":
                if daily_report_enabled and report_scheduler_thread and report_scheduler_thread.is_alive():
                    config_msg = (
                        f"📧 Daily Reports: ENABLED ✅\n"
                        f"⏰ Schedule: {daily_report_time} UTC daily\n"
                        f"📨 From: {sendgrid_from_email}\n"
                        f"📬 To: {', '.join(sendgrid_to_emails)}\n"
                        f"🔑 SendGrid: {'Configured' if sendgrid_api_key else 'Not configured'}"
                    )
                elif daily_report_enabled:
                    config_msg = "📧 Daily Reports: ENABLED but scheduler not running ⚠️"
                else:
                    config_msg = "📧 Daily Reports: DISABLED ❌"
                await websocket.send_json({"role": "bot", "message": config_msg})
                continue
            
            if data.lower() == "/reportstop":
                success = stop_report_scheduler()
                if success:
                    await websocket.send_json({"role": "bot", "message": "🛑 Daily report scheduler stopped."})
                else:
                    await websocket.send_json({"role": "bot", "message": "⚠️ Report scheduler was not running."})
                continue
            
            if data.lower() == "/reportstart":
                success = start_report_scheduler()
                if success:
                    await websocket.send_json({"role": "bot", "message": f"✅ Daily report scheduler started (sends at {daily_report_time} UTC)"})
                else:
                    await websocket.send_json({"role": "bot", "message": "❌ Failed to start report scheduler. Check configuration."})
                continue
            
            if data.lower() == "/reportsend":
                await websocket.send_json({"role": "bot", "message": "📧 Generating and sending security report now..."})
                # Run in background to avoid blocking
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, generate_and_send_daily_report)
                if result:
                    await websocket.send_json({"role": "bot", "message": "✅ Report sent successfully!"})
                else:
                    await websocket.send_json({"role": "bot", "message": "❌ Failed to generate or send report. Check logs."})
                continue
            
            # Regular question
            chat_history.append(HumanMessage(content=data))
            print(f"🧠 Received question: {data}")

            response = qa_chain.invoke({"question": data, "chat_history": chat_history})
            answer = response.get("answer", "").replace("\\n", "\n").strip()
            if not answer:
                answer = "⚠️ Sorry, I couldn't generate a response."

            chat_history.append(SystemMessage(content=answer))
            await websocket.send_json({"role": "bot", "message": answer})

    except WebSocketDisconnect:
        print("⚠️ Client disconnected.")
    except Exception as e:
        print(f"❌ Error in websocket: {e}")
        await websocket.send_json({"role": "bot", "message": f"❌ Error: {str(e)}"})
        await websocket.close()

# ======= HTML UI =======

LOGIN_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Wazuh Chat Assistant - Login</title>
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #1e1e1e;
        color: white;
        margin: 0;
        padding: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .login-container {
        background-color: #252931;
        border: 1px solid #3595F9;
        border-radius: 8px;
        box-shadow: 0 0 10px #3595F9aa;
        padding: 40px;
        width: 350px;
    }
    .login-header {
        text-align: center;
        margin-bottom: 30px;
    }
    .login-header img {
        width: 120px;
        height: auto;
        margin-bottom: 15px;
    }
    .login-header h1 {
        margin: 0;
        font-size: 24px;
        color: #3595F9;
    }
    .login-header p {
        margin: 5px 0 0 0;
        font-size: 14px;
        color: #999;
    }
    .form-group {
        margin-bottom: 20px;
    }
    .form-group label {
        display: block;
        margin-bottom: 8px;
        font-size: 14px;
        color: #ddd;
    }
    .form-group input {
        width: 100%;
        padding: 12px 15px;
        border: 1px solid #3a3f4b;
        border-radius: 6px;
        background-color: #2c2f38;
        color: white;
        font-size: 16px;
        outline: none;
        box-sizing: border-box;
    }
    .form-group input:focus {
        border-color: #3595F9;
    }
    .login-button {
        width: 100%;
        padding: 12px;
        background-color: #3595F9;
        border: none;
        border-radius: 6px;
        color: white;
        font-weight: bold;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.2s ease-in-out;
    }
    .login-button:hover {
        background-color: #1c6dd0;
    }
    .error-message {
        background-color: #ff4444;
        color: white;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 14px;
        display: none;
    }
    .error-message.show {
        display: block;
    }
</style>
</head>
<body>
<div class="login-container">
    <div class="login-header">
        <img src="/logo" alt="Wazuh AI Agent Logo" />
        <h1>Wazuh AI Agent</h1>
        <p>Security Log Analysis Assistant</p>
    </div>
    <div id="error-message" class="error-message"></div>
    <form id="login-form" onsubmit="return handleLogin(event)">
        <div class="form-group">
            <label for="username">Username</label>
            <input type="text" id="username" name="username" required autocomplete="username" autofocus />
        </div>
        <div class="form-group">
            <label for="password">Password</label>
            <input type="password" id="password" name="password" required autocomplete="current-password" />
        </div>
        <button type="submit" class="login-button">Sign In</button>
    </form>
</div>

<script>
    function handleLogin(event) {
        event.preventDefault();
        
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const errorDiv = document.getElementById('error-message');
        
        // Clear previous errors
        errorDiv.classList.remove('show');
        
        // Send login request
        fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        })
        .then(response => {
            if (response.ok) {
                // Redirect to main page on success
                window.location.href = '/';
            } else {
                return response.json();
            }
        })
        .then(data => {
            if (data && data.detail) {
                errorDiv.textContent = data.detail;
                errorDiv.classList.add('show');
            }
        })
        .catch(error => {
            errorDiv.textContent = 'Login failed. Please try again.';
            errorDiv.classList.add('show');
        });
        
        return false;
    }
</script>
</body>
</html>
"""

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Wazuh Chat Assistant</title>
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #1e1e1e;
        color: white;
        margin: 0;
        padding: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 90vh;
        width: 600px;
        max-width: 90vw;
        border: 1px solid #3595F9;
        border-radius: 8px;
        background-color: #252931;
        box-shadow: 0 0 10px #3595F9aa;
    }
    .header {
        padding: 15px 20px;
        background-color: #1e1e1e;
        border-bottom: 1px solid #3595F9;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }
    .header h2 {
        margin: 0;
        font-size: 18px;
        color: #3595F9;
    }
    .logout-btn {
        padding: 8px 16px;
        background-color: #ff4444;
        border: none;
        border-radius: 6px;
        color: white;
        font-size: 14px;
        cursor: pointer;
        transition: background-color 0.2s ease-in-out;
    }
    .logout-btn:hover {
        background-color: #cc0000;
    }
    .messages {
        flex-grow: 1;
        overflow-y: auto;
        padding: 15px;
        display: flex;
        flex-direction: column;
    }
    .message {
        max-width: 70%;
        margin: 5px 0;
        padding: 12px 16px;
        border-radius: 15px;
        word-wrap: break-word;
        white-space: pre-wrap;
        line-height: 1.4;
    }
    .message.user {
        background-color: #3595F9;
        align-self: flex-start;
        color: white;
        border-bottom-left-radius: 0;
    }
    .message.bot {
        background-color: #2c2f38;
        align-self: flex-end;
        color: #ddd;
        border-bottom-right-radius: 0;
    }
    .input-container {
        display: flex;
        padding: 10px 15px;
        background-color: #1e1e1e;
        border-top: 1px solid #3595F9;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
    }
    input[type="text"] {
        flex-grow: 1;
        padding: 12px 15px;
        border: none;
        border-radius: 25px;
        background-color: #2c2f38;
        color: white;
        font-size: 16px;
        outline: none;
    }
    button {
        margin-left: 10px;
        padding: 12px 20px;
        background-color: #3595F9;
        border: none;
        border-radius: 25px;
        color: white;
        font-weight: bold;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.2s ease-in-out;
    }
    button:hover {
        background-color: #1c6dd0;
    }
</style>
</head>
<body>
<div class="chat-container">
    <div class="header">
        <h2>Wazuh AI Assistant</h2>
        <button class="logout-btn" onclick="logout()">Logout</button>
    </div>
    <div class="messages" id="messages"></div>
    <div class="input-container">
        <input type="text" id="user-input" placeholder="Type your message or /help to print the help menu..." autocomplete="off" />
        <button onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
    const messagesDiv = document.getElementById('messages');
    const userInput = document.getElementById('user-input');

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);

    socket.onopen = () => {
        console.log("✅ WebSocket connected");
    };

    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', data.role);
        messageDiv.textContent = data.message;
        messagesDiv.appendChild(messageDiv);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    };

    socket.onclose = () => {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'bot');
        messageDiv.textContent = '⚠️ Connection closed.';
        messagesDiv.appendChild(messageDiv);
    };

    socket.onerror = (error) => {
        console.error("WebSocket error:", error);
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'bot');
        messageDiv.textContent = '⚠️ WebSocket error.';
        messagesDiv.appendChild(messageDiv);
    };

    function sendMessage() {
        const message = userInput.value.trim();
        if (message && socket.readyState === WebSocket.OPEN) {
            // Display user message
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', 'user');
            messageDiv.textContent = message;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            socket.send(message);
            userInput.value = '';
            userInput.focus();
        }
    }

    userInput.addEventListener("keyup", function(event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });
    
    function logout() {
        window.location.href = '/logout';
    }
</script>
</body>
</html>
"""

@app.get("/logo")
async def get_logo():
    logo_path = os.path.join(script_dir, "bot_logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    else:
        # Return 404 if logo not found
        raise HTTPException(status_code=404, detail="Logo not found")

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return LOGIN_PAGE

@app.post("/login")
async def login(request: Request):
    data = await request.json()
    username_input = data.get('username', '')
    password_input = data.get('password', '')
    
    # Verify credentials
    username_match = secrets.compare_digest(username_input, username)
    password_match = secrets.compare_digest(password_input, password)
    
    if username_match and password_match:
        # Set session
        request.session['authenticated'] = True
        request.session['username'] = username_input
        return {"status": "success"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    # Check if user is authenticated
    if not request.session.get('authenticated'):
        return RedirectResponse(url="/login")
    return HTML_PAGE


@app.on_event("startup")
def on_startup():
    print("🚀 Starting FastAPI app and loading vector store...")
    
    # Validate configuration first
    if not validate_required_config():
        print("⚠️  App started but configuration is incomplete")
        print("⚠️  Please configure all required environment variables")
        return
    
    try:
        setup_chain(past_days=days_range)
        print("✅ Startup complete - ready to accept connections")
        
        # Start auto-refresh background thread
        start_auto_refresh()
        
        # Start daily report scheduler if enabled
        start_report_scheduler()
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️  App started but QA chain may not be available")

@app.on_event("shutdown")
def on_shutdown():
    print("👋 Shutting down gracefully...")
    
    # Stop auto-refresh thread
    stop_auto_refresh()
    
    # Stop report scheduler
    stop_report_scheduler()
    
    # Clean up resources
    import gc
    gc.collect()
    
    print("✅ Shutdown complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wazuh AI Agent - Security Log Analysis Assistant")
    parser.add_argument("-d", "--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--opensearch-url", type=str, help="OpenSearch URL (default: https://127.0.0.1:9200)")
    parser.add_argument("--opensearch-user", type=str, help="OpenSearch username (default: admin)")
    parser.add_argument("--opensearch-pass", type=str, help="OpenSearch password")
    parser.add_argument("--opensearch-indexes", type=str, help="Comma-separated OpenSearch index patterns (default: wazuh-alerts-*,wazuh-monitoring-*,wazuh-states-vulnerabilities-*)")
    parser.add_argument("--litellm-url", type=str, help="LiteLLM proxy URL (default: http://localhost:4000)")
    parser.add_argument("--litellm-key", type=str, help="LiteLLM API key")
    parser.add_argument("--model", type=str, help="LiteLLM model to use (default: claude-3-5-sonnet-20241022)")
    parser.add_argument("--auto-refresh-interval", type=int, help="Auto-refresh interval in minutes (default: 15)")
    parser.add_argument("--no-auto-refresh", action="store_true", help="Disable automatic refresh")
    parser.add_argument("--verify-ssl", action="store_true", help="Enable SSL certificate verification")
    args = parser.parse_args()

    # Override configuration with command-line arguments if provided
    if args.opensearch_url:
        opensearch_url = args.opensearch_url
    if args.opensearch_user:
        opensearch_username = args.opensearch_user
    if args.opensearch_pass:
        opensearch_password = args.opensearch_pass
    if args.opensearch_indexes:
        opensearch_indexes = [idx.strip() for idx in args.opensearch_indexes.split(',')]
    if args.litellm_url:
        litellm_api_base = args.litellm_url
    if args.litellm_key:
        litellm_api_key = args.litellm_key
    if args.model:
        litellm_model = args.model
    if args.auto_refresh_interval:
        auto_refresh_interval = args.auto_refresh_interval * 60  # Convert minutes to seconds
    if args.no_auto_refresh:
        auto_refresh_enabled = False
    if args.verify_ssl:
        opensearch_verify_ssl = True

    # Disable SSL warnings for self-signed certificates
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Set multiprocessing start method to avoid issues on macOS
    try:
        import multiprocessing
        # Only set if not already set (avoid conflicts on Linux)
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # Already set
    
    try:
        if args.daemon:
            run_daemon()
        else:
            uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()