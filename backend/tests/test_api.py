import sys
from types import ModuleType
from pathlib import Path
from importlib import import_module
from fastapi.testclient import TestClient

# Provide dummy mlx_whisper to satisfy imports if not installed
if 'mlx_whisper' not in sys.modules:
    sys.modules['mlx_whisper'] = ModuleType('mlx_whisper')

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "backend" / "app"))

app_module = import_module("app")
app = app_module.app


def test_get_models():
    with TestClient(app) as client:
        response = client.get("/api/models")
        assert response.status_code == 200
        assert isinstance(response.json(), dict)


def test_status_returns_list():
    with TestClient(app) as client:
        response = client.get("/api/status")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
