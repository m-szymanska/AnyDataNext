import os
import uuid
from pathlib import Path
from backend.app.utils.anonymizer import detect_pii, anonymize_text
from backend.app.utils.progress import save_progress, get_progress, PROGRESS_DIR


def test_detect_pii_basic():
    text = "Kontakt: jan.kowalski@example.com oraz +48 123-456-789."
    findings = detect_pii(text)
    assert "email" in findings
    assert "phone" in findings


def test_anonymize_text_replaces_pii():
    text = "Email: user@example.com"
    anonymized = anonymize_text(text)
    assert "[EMAIL" in anonymized


def test_save_and_get_progress():
    job_id = f"test_{uuid.uuid4().hex}"
    progress_file = PROGRESS_DIR / f"{job_id}_progress.json"
    try:
        save_progress(job_id, 10, processed=5)
        data = get_progress(job_id)
        assert data["processed"] == 5
        assert progress_file.is_file()
    finally:
        if progress_file.exists():
            progress_file.unlink()
