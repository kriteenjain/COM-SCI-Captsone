"""GCS checkpoint and metrics storage for distributed training.

Uses Application Default Credentials, which are automatic on GCP VMs
(no key files needed). Set GCS_BUCKET env var to enable.
"""

import glob
import os
import tarfile
import tempfile
from typing import Optional

from google.cloud import storage


def _get_client() -> storage.Client:
    return storage.Client()


def upload_checkpoint(local_dir: str, bucket_name: str, prefix: str) -> None:
    """Tar checkpoint files and upload to GCS.

    Uploads all ckpt-* and related files from local_dir to
    gs://<bucket_name>/<prefix>/checkpoint.tar.gz
    """
    client = _get_client()
    bucket = client.bucket(bucket_name)

    patterns = ["ckpt-*", "checkpoint", "cumulative_wall_time"]
    files_to_upload = []
    for pattern in patterns:
        files_to_upload.extend(glob.glob(os.path.join(local_dir, pattern)))

    if not files_to_upload:
        return

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            for fpath in files_to_upload:
                tar.add(fpath, arcname=os.path.basename(fpath))

        blob = bucket.blob(f"{prefix}/checkpoint.tar.gz")
        blob.upload_from_filename(tmp_path)
    finally:
        os.unlink(tmp_path)


def download_latest_checkpoint(
    bucket_name: str, prefix: str, local_dir: str
) -> Optional[str]:
    """Download the latest checkpoint tarball from GCS and extract to local_dir.

    Returns the local directory path if a checkpoint was downloaded, None otherwise.
    """
    client = _get_client()
    bucket = client.bucket(bucket_name)

    blob = bucket.blob(f"{prefix}/checkpoint.tar.gz")
    if not blob.exists():
        print(f"[gcs] No checkpoint found at gs://{bucket_name}/{prefix}/checkpoint.tar.gz")
        return None

    os.makedirs(local_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        blob.download_to_filename(tmp_path)
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=local_dir)
        print(f"[gcs] Checkpoint extracted to {local_dir}")
        return local_dir
    finally:
        os.unlink(tmp_path)


def upload_file(local_path: str, bucket_name: str, remote_path: str) -> None:
    """Upload a single file to GCS."""
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_path)


def download_file(
    bucket_name: str, remote_path: str, local_path: str
) -> Optional[str]:
    """Download a single file from GCS. Returns local_path on success, None if not found."""
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(remote_path)

    if not blob.exists():
        return None

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path
