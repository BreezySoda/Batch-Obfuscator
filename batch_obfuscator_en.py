# -*- coding: utf-8 -*-
"""
Batch obfuscator with token-aware handling of:
- labels (:label)
- variable references (%VAR%), delayed !VAR!
- command operators: &&, ||, &, | 
- redirects: >, >>, 2>, 2>>, < etc.
"""

from __future__ import annotations
import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ALPHABET = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ @=")
KEY_PREFIX = "VAR"
NUM_KEYS = 5

OPERATORS = ["&&", "||", ">>", "2>>", "2>", "1>", "1>>", ">|", "&", "|", ">", "<", ";"]


def clean_comments(content: str) -> str:
    out_lines: List[str] = []
    for ln in content.splitlines():
        stripped = ln.lstrip()
        low = stripped.lower()
        if low.startswith("rem ") or low == "rem" or low.startswith("::"):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines)


def generate_substrings(num_keys: int = NUM_KEYS, alphabet: List[str] = ALPHABET) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    base = alphabet[:]
    for i in range(num_keys):
        pool = base[:]
        random.shuffle(pool)
        mapping[f"{KEY_PREFIX}{i}"] = "".join(pool)
    return mapping


def tokenize_line(line: str) -> List[Tuple[str, str]]:
    """
    Tokenize a line into list of (type, text).
    Types: 'label', 'varref', 'delayed', 'op', 'text'
    """
    tokens: List[Tuple[str, str]] = []
    i = 0
    if line.startswith(":"):
        return [("label", line)]
    while i < len(line):
        ch = line[i]
        if ch == "%":
            j = i + 1
            while j < len(line) and line[j] != "%":
                j += 1
            if j < len(line):
                tokens.append(("varref", line[i:j+1]))
                i = j + 1
                continue
            else:
                tokens.append(("text", ch))
                i += 1
                continue
        if ch == "!":
            j = i + 1
            while j < len(line) and line[j] != "!":
                j += 1
            if j < len(line):
                tokens.append(("delayed", line[i:j+1]))
                i = j + 1
                continue
            else:
                tokens.append(("text", ch))
                i += 1
                continue
        matched = False
        for op in OPERATORS:
            if line.startswith(op, i):
                tokens.append(("op", op))
                i += len(op)
                matched = True
                break
        if matched:
            continue
        j = i
        while j < len(line) and line[j] not in "%!":
            if any(line.startswith(op, j) for op in OPERATORS):
                break
            j += 1
        tokens.append(("text", line[i:j]))
        i = j
    return tokens


def obfuscate_with_substrings(content: str, mapping: Dict[str, str]) -> str:
    keys = list(mapping.keys())
    values = [mapping[k] for k in keys]
    out_lines: List[str] = []

    for line in content.splitlines():
        if line.startswith(":"):
            out_lines.append(line)
            continue
        toks = tokenize_line(line)
        new_parts: List[str] = []
        for ttype, txt in toks:
            if ttype in ("label", "varref", "delayed", "op"):
                new_parts.append(txt)
                continue
            buf: List[str] = []
            for ch in txt:
                replaced = False
                for idx, val in enumerate(values):
                    pos = val.find(ch)
                    if pos != -1:
                        buf.append(f"%{keys[idx]}:~{pos},1%")
                        replaced = True
                        break
                if not replaced:
                    buf.append(ch)
            new_parts.append("".join(buf))
        out_lines.append("".join(new_parts))
    return "\n".join(out_lines)


def generate_set_lines(mapping: Dict[str, str]) -> List[str]:
    return [f'set "{k}={v}"' for k, v in mapping.items()]


def backup_if_exists(path: Path) -> None:
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)


def process_file(path: Path, num_keys: int = NUM_KEYS) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_comments(raw)
    mapping = generate_substrings(num_keys=num_keys)
    set_lines = generate_set_lines(mapping)
    body = obfuscate_with_substrings(cleaned, mapping)
    out = "@echo off\n" + "\n".join(set_lines) + "\n" + body + "\n"
    out_path = path.with_name(path.stem + "-obf" + path.suffix)
    backup_if_exists(out_path)
    out_path.write_text(out, encoding="utf-8")
    return out_path


def cli() -> int:
    p = argparse.ArgumentParser(description="Batch (.bat) obfuscator (handles &&, ||, >, |, etc.)")
    p.add_argument("file", type=Path, help="Path to .bat file")
    p.add_argument("--keys", type=int, default=NUM_KEYS, help="Number of VAR keys to generate")
    args = p.parse_args()

    try:
        out = process_file(args.file, num_keys=args.keys)
        print("Obfuscated Batch written to:", out)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
