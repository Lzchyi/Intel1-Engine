from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .output_contract import REQUIRED_FILES, manifest_payload, validate_outputs, write_json
from .structured_data import WeekendContext


def upload_bundle(
    *,
    output_dir: Path,
    run_id: str,
    weekend: WeekendContext,
    folder_id: str,
    service_account_json: str,
    make_public: bool,
) -> str:
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as error:
        raise RuntimeError("Google Drive upload requires requirements.txt dependencies.") from error

    info = json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)

    file_urls: dict[str, str] = {}
    for name in REQUIRED_FILES:
        file_id = upsert_file(service, output_dir / name, name, folder_id, make_public)
        file_urls[name] = direct_download_url(file_id)

    write_json(output_dir / "app_manifest.json", manifest_payload(run_id, weekend, output_dir, file_urls=file_urls))
    validate_outputs(output_dir)
    manifest_id = upsert_file(service, output_dir / "app_manifest.json", "app_manifest.json", folder_id, make_public)
    return direct_download_url(manifest_id)


def upsert_file(service: Any, path: Path, name: str, folder_id: str, make_public: bool) -> str:
    existing = find_file(service, name, folder_id)
    media = media_upload(path)
    metadata = {"name": name, "mimeType": "application/json"}
    if existing:
        file_id = existing["id"]
        service.files().update(fileId=file_id, body=metadata, media_body=media, fields="id").execute()
    else:
        metadata["parents"] = [folder_id]
        created = service.files().create(body=metadata, media_body=media, fields="id").execute()
        file_id = created["id"]
    if make_public:
        ensure_public(service, file_id)
    return file_id


def find_file(service: Any, name: str, folder_id: str) -> dict[str, str] | None:
    escaped_name = name.replace("'", "\\'")
    query = f"name = '{escaped_name}' and '{folder_id}' in parents and trashed = false"
    result = service.files().list(q=query, spaces="drive", fields="files(id,name)", pageSize=10).execute()
    files = result.get("files", [])
    return files[0] if files else None


def media_upload(path: Path) -> Any:
    from googleapiclient.http import MediaFileUpload

    return MediaFileUpload(str(path), mimetype="application/json", resumable=False)


def ensure_public(service: Any, file_id: str) -> None:
    try:
        service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
            fields="id",
        ).execute()
    except Exception:
        # Existing permission or domain policy can make this non-fatal; upload still succeeded.
        pass


def direct_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"
