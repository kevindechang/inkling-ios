#!/usr/bin/env python3
"""SageMaker entry for distill_v2_sota.py (WSD-40 v2: clean-teacher, QAT-base student, 128/128 experts)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

CKPT = Path("/opt/ml/checkpoints")
HERE = Path(__file__).resolve().parent
LOG = CKPT / "log.txt"
SNAPSHOT_S = 45
MAX_OOM_RETRIES = 2
FLAG = "__flag__"
LLAMA_CPP_DIR = "/opt/ml/code/llama.cpp"
STUDENT_GGUF = "/tmp/student_base.gguf"


def run(cmd: list[str], log) -> int:
    log.write(f"$ {' '.join(cmd)}\n")
    log.flush()
    return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)


def write_snapshot() -> None:
    try:
        if LOG.is_file():
            tmp = CKPT / "log_snapshot.txt.tmp"
            shutil.copyfile(LOG, tmp)
            os.replace(tmp, CKPT / "log_snapshot.txt")
    except OSError:
        pass


def snapshot_log() -> None:
    while True:
        time.sleep(SNAPSHOT_S)
        write_snapshot()


def batch_of(args: list[str]) -> int:
    hits = [int(args[i + 1]) for i, a in enumerate(args[:-1]) if a == "--batch"]
    return hits[-1] if hits else 2


def halve_batch(args: list[str]) -> list[str]:
    def last(flag: str, default: int) -> int:
        hits = [int(args[i + 1]) for i, a in enumerate(args[:-1]) if a == flag]
        return hits[-1] if hits else default
    return args + ["--batch", str(max(1, last("--batch", 2) // 2)), "--grad-accum", str(last("--grad-accum", 4) * 2)]


def split_student_gguf(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise SystemExit(f"--student-gguf-hf wants repo_id:filename, got {spec!r}")
    repo, filename = spec.rsplit(":", 1)
    if not repo or not filename:
        raise SystemExit(f"--student-gguf-hf wants repo_id:filename, got {spec!r}")
    return repo, filename


def main() -> int:
    CKPT.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=snapshot_log, daemon=True).start()
    os.environ.setdefault("HF_HOME", "/opt/ml/hf")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with open(LOG, "a", buffering=1) as log:
        try:
            run(["nvidia-smi"], log)
            run(["free", "-g"], log)
            run(["df", "-h", "/opt/ml", "/tmp"], log)
            if run([
                sys.executable, "-m", "pip", "install", "-q",
                "transformers==5.24.0", "datasets", "pyyaml", "sentencepiece", "tiktoken",
                "hf_transfer", "awscli", "huggingface_hub",
            ], log):
                log.write("pip install failed\n")
                return 1
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            run([
                sys.executable, "-c",
                "import torch, transformers, datasets, huggingface_hub; "
                "print(torch.__version__, transformers.__version__, datasets.__version__, huggingface_hub.__version__)",
            ], log)
            raw_args = [a for a in sys.argv[1:] if a != FLAG]
            clean_args: list[str] = []
            i = 0
            while i < len(raw_args):
                flag = raw_args[i]
                if flag == "--init-from-s3" and i + 1 < len(raw_args):
                    uri = raw_args[i + 1]
                    dst = "/tmp/init_ckpt.pt"
                    if not Path(dst).is_file():
                        if run(["aws", "s3", "cp", uri, dst, "--region", "us-east-1", "--only-show-errors"], log):
                            log.write(f"failed to download init ckpt {uri}\n")
                            return 1
                    clean_args += ["--init-from", dst]
                    i += 2
                elif flag == "--student-gguf-hf" and i + 1 < len(raw_args):
                    repo, filename = split_student_gguf(raw_args[i + 1])
                    if not Path(STUDENT_GGUF).is_file():
                        from huggingface_hub import hf_hub_download
                        t_dl = time.time()
                        log.write(f"downloading student GGUF {repo}:{filename}\n")
                        log.flush()
                        src = hf_hub_download(repo_id=repo, filename=filename)
                        if os.path.lexists(STUDENT_GGUF):
                            os.unlink(STUDENT_GGUF)
                        os.symlink(src, STUDENT_GGUF)
                        log.write(
                            f"student GGUF ready at {STUDENT_GGUF} -> {src} "
                            f"({Path(src).stat().st_size} bytes in {time.time() - t_dl:.1f}s)\n"
                        )
                    clean_args += [
                        "--student-from-gguf", STUDENT_GGUF,
                        "--llama-cpp-dir", LLAMA_CPP_DIR,
                    ]
                    i += 2
                else:
                    clean_args.append(flag)
                    i += 1
            args = [
                sys.executable, str(HERE / "distill_v2_sota.py"),
                "--hot-plan", str(HERE / "hot_plan_40.json"),
                "--out", str(CKPT / "run"),
                "--resume",
            ] + clean_args
            mark = LOG.stat().st_size
            code = run(args, log)
            log.write(f"distill_v2_sota.py exited {code}\n")
            for _ in range(MAX_OOM_RETRIES):
                with open(LOG, "rb") as f:
                    f.seek(mark)
                    attempt = f.read().decode(errors="replace").lower()
                if code == 0 or "out of memory" not in attempt or batch_of(args) <= 1:
                    break
                args = halve_batch(args)
                log.write(f"CUDA OOM: retrying with {' '.join(args[-4:])}\n")
                log.flush()
                mark = LOG.stat().st_size
                code = run(args, log)
                log.write(f"distill_v2_sota.py exited {code}\n")
            run(["nvidia-smi"], log)
            return code
        except Exception:
            log.write(traceback.format_exc())
            return 1
        finally:
            log.flush()
            write_snapshot()


if __name__ == "__main__":
    sys.exit(main())
