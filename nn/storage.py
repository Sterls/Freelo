"""
rclone-based sync to/from Google Drive.

Setup (one-time):
    rclone config   # create a remote, name it 'gdrive', choose Google Drive

The Drive folder ID comes from the URL:
  https://drive.google.com/drive/folders/<FOLDER_ID>

Override via env vars:
    RCLONE_REMOTE     — rclone remote name   (default: gdrive)
    RCLONE_FOLDER_ID  — Drive folder ID      (default: hardcoded below)
"""

import os
import shutil
import subprocess

REMOTE    = os.getenv("RCLONE_REMOTE",    "gdrive")
FOLDER_ID = os.getenv("RCLONE_FOLDER_ID", "1NmRIdN4m3sipCXAG4JqxahCjKMYO3Yha")


def _available() -> bool:
    return shutil.which("rclone") is not None


def _run(*args: str):
    cmd = ["rclone", *args, "--drive-root-folder-id", FOLDER_ID]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"rclone error:\n{result.stderr.strip()}")
    return result.stdout


def push(local_dir: str, remote_subdir: str = None):
    """Copy new/updated files from local_dir to Drive/<remote_subdir>.
    Uses 'copy' (not 'sync') so files pruned locally are never deleted on Drive.
    """
    if not _available():
        print("  [storage] rclone not found, skipping push")
        return
    subdir = remote_subdir or os.path.basename(local_dir.rstrip("/\\"))
    _run("copy", local_dir, f"{REMOTE}:{subdir}", "--progress")
    print(f"  [storage] pushed {local_dir} → {REMOTE}:{subdir}")


def pull(local_dir: str, remote_subdir: str = None):
    """Sync Drive/<remote_subdir> down to local_dir."""
    if not _available():
        print("  [storage] rclone not found, skipping pull")
        return
    subdir = remote_subdir or os.path.basename(local_dir.rstrip("/\\"))
    os.makedirs(local_dir, exist_ok=True)
    try:
        _run("copy", f"{REMOTE}:{subdir}", local_dir, "--progress")
        print(f"  [storage] pulled {REMOTE}:{subdir} → {local_dir}")
    except RuntimeError as e:
        if "not found" in str(e).lower():
            print(f"  [storage] {REMOTE}:{subdir} doesn't exist yet, skipping pull")
        else:
            raise
