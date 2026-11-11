import base64
import io
from fastapi.testclient import TestClient
from pathlib import Path

from ..app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# Simple 1x1 PNG (white) base64
PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/\n"
    "w8AAn8B9W7qwAAAABJRU5ErkJggg=="
)

image_bytes = base64.b64decode(PNG_BASE64)

def test_classify_unknown():
    files = {"file": ("test.png", image_bytes, "image/png")}
    response = client.post("/api/v1/classify?language=en", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["category"] in ["Biodegradable", "Recyclable", "E-waste", "Hazardous", "Unknown"]

def test_history_and_delete():
    # ensure history list works
    resp = client.get("/api/v1/history")
    assert resp.status_code == 200
    # delete a non-existing id returns 404
    del_resp = client.delete("/api/v1/history/nonexistent-id")
    assert del_resp.status_code == 404

