"""MongoDB Atlas upload with a local SQLite queue, so events survive internet drops.

Set the connection string first:
    Windows PowerShell:  $env:MONGODB_URI="mongodb+srv://user:pass@cluster/..."
    Linux / Pi:          export MONGODB_URI="mongodb+srv://..."
"""
import datetime as dt
import json
import os
import sqlite3

QUEUE_DB = os.environ.get("QUEUE_DB", "event_queue.sqlite")
DB_NAME = os.environ.get("MONGODB_DB", "broiler_monitoring")
COLLECTION = "events"

_conn = sqlite3.connect(QUEUE_DB)
_conn.execute("CREATE TABLE IF NOT EXISTS queue (id INTEGER PRIMARY KEY, doc TEXT)")
_conn.commit()
_client = None


def _collection():
    global _client
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        return None
    if _client is None:
        from pymongo import MongoClient
        _client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    return _client[DB_NAME][COLLECTION]


def log_event(event_type, payload=None, **extra):
    """Queue an event locally, then try to flush the queue to the cloud."""
    doc = {"type": event_type, "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
           **(payload or {}), **extra}
    _conn.execute("INSERT INTO queue (doc) VALUES (?)", (json.dumps(doc),))
    _conn.commit()
    return flush()


def flush(batch=500):
    """Send queued events. Returns True if the queue is empty afterwards."""
    col = _collection()
    if col is None:
        return False
    try:
        while True:
            rows = _conn.execute("SELECT id, doc FROM queue ORDER BY id LIMIT ?", (batch,)).fetchall()
            if not rows:
                return True
            docs = []
            for _, raw in rows:
                d = json.loads(raw)
                d["ts"] = dt.datetime.fromisoformat(d["ts"])
                docs.append(d)
            col.insert_many(docs)
            _conn.executemany("DELETE FROM queue WHERE id = ?", [(r[0],) for r in rows])
            _conn.commit()
    except Exception as e:          # offline, auth error, etc.: keep events queued
        print(f"[cloud] flush failed, {pending()} event(s) kept in queue: {e}")
        return False


def pending():
    return _conn.execute("SELECT COUNT(*) FROM queue").fetchone()[0]