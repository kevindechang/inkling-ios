#!/usr/bin/env python3
"""Build a 100% leak-free, multi-domain calibration & holdout corpus for Mac FMRD + EAGLE training.

Zero pip dependencies (uses HuggingFace Datasets Server REST API directly for math & tool-calling!).
Mix (interleaved round-robin so strided sampling hits all domains evenly):
  - 45% On-policy Inkling-2 rollouts (rollouts-a3-clean.jsonl + rollouts-a3c.jsonl + a1-clean, general + companion + thinking)
  - 20% GSM8K train (openai/gsm8k train, formatted with Inkling-2 <|content_thinking|>)
  - 20% MATH train (EleutherAI/hendrycks_math train across 7 subsets, formatted with <|content_thinking|>)
  - 15% Tool-calling (NousResearch/hermes-function-calling-v1, formatted with Inkling-2 tool/chat tokens)

Strict contamination guard:
  - Filters EVERY candidate against eval/exclude_eval_prompts.jsonl
    (exact + whitespace-normalized SHA-256 of user prompts).
  - Splits 90% train (/tmp/fmrd_calib.jsonl) and 10% disjoint holdout (/tmp/fmrd_holdout.jsonl).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(".")
EXCLUDE_PATH = Path("eval/exclude_eval_prompts.jsonl")
OUT_TRAIN = Path("/tmp/fmrd_calib.jsonl")
OUT_HOLD = Path("/tmp/fmrd_holdout.jsonl")

END_MESSAGE = "<|end_message|>"
MESSAGE_MODEL = "<|message_model|>"
CALC_ANN_RE = re.compile(r"<<[^<>]*>>")
WS_RE = re.compile(r"\s+")
USER_BLOCK_RE = re.compile(r"<\|message_user\|><\|content_text\|>(.*?)<\|end_message\|>", re.DOTALL)

MATH_SUBSETS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)


def norm_text(s: str) -> str:
    return WS_RE.sub(" ", s.strip().lower())


def load_exclusions() -> set[str]:
    hashes = set()
    for line in EXCLUDE_PATH.open():
        r = json.loads(line)
        for k in ("sha256_all_user", "sha256_first_user", "sha256_last_user"):
            v = r.get(k)
            if v:
                hashes.add(v)
    return hashes


def is_excluded(text: str, hashes: set[str]) -> bool:
    users = USER_BLOCK_RE.findall(text)
    if not users:
        return False
    all_u = "\n".join(u.strip() for u in users)
    for cand in (all_u, users[0].strip(), users[-1].strip()):
        if hashlib.sha256(cand.encode("utf-8")).hexdigest() in hashes:
            return True
        if hashlib.sha256(norm_text(cand).encode("utf-8")).hexdigest() in hashes:
            return True
    return False


def hf_fetch_rows(dataset: str, config: str, split: str, n_rows: int, offset: int = 0) -> list[dict]:
    out = []
    pos = offset
    while len(out) < n_rows:
        take = min(100, n_rows - len(out))
        qs = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": pos, "length": take})
        url = f"https://datasets-server.huggingface.co/rows?{qs}"
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "inkling-ios-calib/1.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read())
                rows = [r["row"] for r in data.get("rows", [])]
                out.extend(rows)
                if len(rows) < take:
                    return out
                break
            except Exception as err:
                if attempt == 3:
                    print(f"[warn] hf_fetch_rows({dataset}/{config}@{pos}) failed: {err}")
                    return out
                time.sleep(1.5 * (attempt + 1))
        pos += take
    return out


def extract_boxed(sol: str) -> str:
    idx = sol.rfind("\\boxed{")
    if idx < 0:
        return sol.strip().splitlines()[-1].strip()
    depth, out = 0, []
    for ch in sol[idx + 7:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                break
            depth -= 1
        out.append(ch)
    return "".join(out).strip()


def load_onpolicy(rng: random.Random) -> list[dict]:
    rows = []
    for fn in ("rollouts-a3-clean.jsonl", "rollouts-a3c.jsonl", "rollouts-a1-clean.jsonl"):
        p = ROOT / "train/onpolicy" / fn
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            txt = r.get("text", "")
            if len(txt) >= 120:
                rows.append({"domain": f"onpolicy_{r.get('kind', 'gen')}", "text": txt})
    rng.shuffle(rows)
    return rows


def load_hf_gsm8k(rng: random.Random, limit: int = 240) -> list[dict]:
    raw = hf_fetch_rows("openai/gsm8k", "main", "train", limit, offset=0)
    rng.shuffle(raw)
    out = []
    for r in raw:
        q = str(r.get("question", "")).strip()
        a = str(r.get("answer", "")).strip()
        if not q or "####" not in a:
            continue
        steps, final = a.rsplit("####", 1)
        steps = CALC_ANN_RE.sub("", steps).strip()
        final = final.strip()
        if not steps or not final:
            continue
        txt = (
            f"<|message_system|><|content_text|>Thinking effort level: 0.9{END_MESSAGE}"
            f"<|message_user|><|content_text|>{q}\n\nEnd with: Answer: <final answer as a number>{END_MESSAGE}"
            f"{MESSAGE_MODEL}<|content_thinking|>{steps}{END_MESSAGE}"
            f"{MESSAGE_MODEL}<|content_text|>Answer: {final}{END_MESSAGE}"
        )
        out.append({"domain": "gsm8k_train", "text": txt})
    return out


def load_hf_math(rng: random.Random, per_sub: int = 35) -> list[dict]:
    out = []
    for sub in MATH_SUBSETS:
        raw = hf_fetch_rows("EleutherAI/hendrycks_math", sub, "train", per_sub + 10, offset=0)
        added = 0
        for r in raw:
            prob = str(r.get("problem", "")).strip()
            sol = str(r.get("solution", "")).strip()
            if not prob or not sol or len(sol) > 3500:
                continue
            ans = extract_boxed(sol)
            txt = (
                f"<|message_system|><|content_text|>Thinking effort level: 0.9{END_MESSAGE}"
                f"<|message_user|><|content_text|>{prob}{END_MESSAGE}"
                f"{MESSAGE_MODEL}<|content_thinking|>{sol}{END_MESSAGE}"
                f"{MESSAGE_MODEL}<|content_text|>Answer: {ans}{END_MESSAGE}"
            )
            out.append({"domain": "math_train", "text": txt})
            added += 1
            if added >= per_sub:
                break
    rng.shuffle(out)
    return out


def load_hf_tools(rng: random.Random, limit: int = 200) -> list[dict]:
    raw = []
    raw.extend(hf_fetch_rows("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", 140, offset=0))
    raw.extend(hf_fetch_rows("NousResearch/hermes-function-calling-v1", "glaive_func_calling", "train", 100, offset=0))
    rng.shuffle(raw)
    out = []
    for r in raw:
        convs = r.get("conversations") or []
        if len(convs) < 2:
            continue
        sys_parts = ["Thinking effort level: 0.0"]
        turns = []
        for m in convs:
            fr = str(m.get("from", "")).lower()
            val = str(m.get("value", "")).strip()
            if not val:
                continue
            if fr == "system":
                sys_parts.append(val)
            elif fr in ("human", "user"):
                turns.append(f"<|message_user|><|content_text|>{val}{END_MESSAGE}")
            elif fr in ("gpt", "assistant"):
                turns.append(f"{MESSAGE_MODEL}<|content_text|>{val}{END_MESSAGE}")
            elif fr == "tool":
                turns.append(f"<|message_tool|><|content_text|>{val}{END_MESSAGE}")
        if not any(t.startswith(MESSAGE_MODEL) for t in turns):
            continue
        txt = f"<|message_system|><|content_text|>{chr(10).join(sys_parts)}{END_MESSAGE}" + "".join(turns)
        if 150 <= len(txt) <= 4500:
            out.append({"domain": "tool_train", "text": txt})
        if len(out) >= limit:
            break
    return out


def main() -> None:
    rng = random.Random(20261127)
    hashes = load_exclusions()
    print(f"[calib] loaded {len(hashes)} eval prompt SHA-256 hashes from {EXCLUDE_PATH.name}", flush=True)

    buckets = {
        "onpolicy": load_onpolicy(rng),
        "gsm8k": load_hf_gsm8k(rng, limit=240),
        "math": load_hf_math(rng, per_sub=35),
        "tools": load_hf_tools(rng, limit=200),
    }

    filtered: dict[str, list[dict]] = {}
    dropped = 0
    for k, rows in buckets.items():
        keep = [r for r in rows if not is_excluded(r["text"], hashes)]
        dropped += len(rows) - len(keep)
        filtered[k] = keep
        print(f"  bucket {k:10s}: kept={len(keep):4d} (raw={len(rows):4d})", flush=True)
    print(f"[calib] contamination filter dropped {dropped} rows", flush=True)

    train_b, hold_b = {}, {}
    for k, rows in filtered.items():
        n_hold = max(1, len(rows) // 10) if len(rows) >= 10 else 0
        hold_b[k] = rows[:n_hold]
        train_b[k] = rows[n_hold:]

    def interleave(b: dict[str, list[dict]]) -> list[dict]:
        pattern = ["onpolicy"] * 8 + ["gsm8k"] * 4 + ["math"] * 4 + ["tools"] * 4
        idx = {k: 0 for k in b}
        res = []
        while True:
            progressed = False
            for k in pattern:
                if idx[k] < len(b[k]):
                    res.append(b[k][idx[k]])
                    idx[k] += 1
                    progressed = True
            if not progressed:
                break
        return res

    train_rows = interleave(train_b)
    hold_rows = interleave(hold_b)

    with OUT_TRAIN.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_HOLD.open("w") as f:
        for r in hold_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[calib] WROTE train={len(train_rows)} ({OUT_TRAIN.stat().st_size/1e6:.2f} MB) "
          f"holdout={len(hold_rows)} ({OUT_HOLD.stat().st_size/1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
