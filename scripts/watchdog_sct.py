"""Auto-restart watchdog for sCT U-Net (full_v2) training.

Resumes from outputs/full_v2/last.pth on crash. Stops when 'Training done.'
appears in train.log or `last.pth` shows epoch == NUM_EPOCHS-1.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:/Users/USER/Desktop/dev/sct_unet")
PY = Path(r"C:/Users/USER/anaconda3/envs/nnunet/python.exe")
CONFIG = ROOT / "configs" / "full_v2.yaml"
OUT_DIR = ROOT / "outputs" / "full_v2"
LAST_CKPT = OUT_DIR / "last.pth"
TRAIN_LOG = OUT_DIR / "train.log"

LOG_DIR = ROOT / "scripts" / "watchdog_v2_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
WATCHDOG_LOG = LOG_DIR / "watchdog.log"
STATE_FILE = LOG_DIR / "watchdog_state.txt"

NUM_EPOCHS = 1000
MAX_ATTEMPTS = 100
BACKOFF_SECS = 30
HARD_FAIL_THRESHOLD = 8


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_state(payload: dict) -> None:
    STATE_FILE.write_text(
        "\n".join(f"{k}={v}" for k, v in payload.items()), encoding="utf-8"
    )


def latest_epoch() -> int:
    if not TRAIN_LOG.exists():
        return -1
    last = -1
    pat = re.compile(r"epoch (\d+) val MAE")
    try:
        with open(TRAIN_LOG, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                m = pat.search(ln)
                if m:
                    last = max(last, int(m.group(1)))
    except OSError:
        pass
    return last


def training_complete() -> bool:
    if not TRAIN_LOG.exists():
        return False
    try:
        with open(TRAIN_LOG, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if "Training done." in ln:
                    return True
    except OSError:
        pass
    return False


def run_once(attempt: int) -> int:
    cmd = [str(PY), "-m", "src.train", "--config", str(CONFIG)]
    if LAST_CKPT.exists():
        cmd.extend(["--resume", str(LAST_CKPT)])
        log(f"attempt {attempt}: resuming from {LAST_CKPT.name}")
    else:
        log(f"attempt {attempt}: fresh start (no last.pth)")
    out_path = LOG_DIR / f"attempt_{attempt:03d}.log"
    with open(out_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def main() -> int:
    log("watchdog_sct (full_v2) started")
    consecutive_fails = 0
    last_seen_epoch = latest_epoch()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        write_state({
            "status": "running",
            "attempt": attempt,
            "consecutive_fails": consecutive_fails,
            "last_seen_epoch": last_seen_epoch,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        try:
            code = run_once(attempt)
        except Exception as e:
            log(f"attempt {attempt}: launcher exception: {e!r}")
            code = -999
        new_epoch = latest_epoch()
        progress = new_epoch - last_seen_epoch if last_seen_epoch >= 0 else new_epoch
        log(
            f"attempt {attempt}: exit={code} epoch_before={last_seen_epoch} "
            f"epoch_after={new_epoch} progress={progress}"
        )
        if training_complete() or new_epoch >= NUM_EPOCHS - 1:
            log(f"training complete (epoch={new_epoch}); stopping watchdog")
            write_state({
                "status": "done",
                "attempt": attempt,
                "last_seen_epoch": new_epoch,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            return 0
        if progress > 0:
            consecutive_fails = 0
        else:
            consecutive_fails += 1
        last_seen_epoch = new_epoch
        if consecutive_fails >= HARD_FAIL_THRESHOLD:
            log(f"consecutive_fails={consecutive_fails} without progress; giving up")
            write_state({
                "status": "gave_up",
                "attempt": attempt,
                "last_seen_epoch": new_epoch,
                "consecutive_fails": consecutive_fails,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            return 2
        log(f"sleeping {BACKOFF_SECS}s before next attempt")
        time.sleep(BACKOFF_SECS)
    log("max attempts exhausted")
    return 3


if __name__ == "__main__":
    sys.exit(main())
