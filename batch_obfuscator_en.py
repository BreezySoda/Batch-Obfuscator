from __future__ import annotations
import argparse
import logging
import random
import re
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ALPHABET = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ @=")
KEY_PREFIX = "VAR"
NUM_KEYS = 5

OPERATORS = ["&&", "||", ">>", "2>>", "2>", "1>", "1>>", ">|", "&", "|", ">", "<", ";"]
OPERATORS.sort(key=len, reverse=True)

LABEL_PATTERN = re.compile(r"^:")
VARREF_PATTERN = re.compile(r"%[^%]*%")
DELAYED_PATTERN = re.compile(r"![^!]*!")


class TokenType(Enum):
    LABEL = "label"
    VARREF = "varref"
    DELAYED = "delayed"
    OPERATOR = "op"
    TEXT = "text"


@dataclass
class Token:
    type: TokenType
    text: str

    def __repr__(self) -> str:
        return f"Token({self.type.value}: {self.text!r})"


class ObfuscationError(Exception):
    pass


class InvalidFileError(ObfuscationError):
    pass


class MappingError(ObfuscationError):
    pass


def clean_comments(content: str) -> str:
    out_lines: List[str] = []
    for ln in content.splitlines():
        stripped = ln.lstrip()
        low = stripped.lower()
        if low.startswith("rem ") or low == "rem" or low.startswith("::"):
            continue
        out_lines.append(ln)
    
    logger.debug(f"Cleaned comments: {len(content.splitlines())} -> {len(out_lines)} lines")
    return "\n".join(out_lines)


def generate_substrings(num_keys: int = NUM_KEYS, alphabet: List[str] | None = None) -> Dict[str, str]:
    if alphabet is None:
        alphabet = ALPHABET.copy()
    
    if num_keys <= 0:
        raise MappingError(f"num_keys must be positive, got {num_keys}")
    
    if num_keys > 100:
        raise MappingError(f"num_keys too large: {num_keys} (max 100)")
    
    mapping: Dict[str, str] = {}
    for i in range(num_keys):
        pool = alphabet.copy()
        random.shuffle(pool)
        mapping[f"{KEY_PREFIX}{i}"] = "".join(pool)
    
    logger.debug(f"Generated {num_keys} variable mappings")
    return mapping


def tokenize_line(line: str) -> List[Token]:
    if line.startswith(":"):
        return [Token(TokenType.LABEL, line)]
    
    tokens: List[Token] = []
    i = 0
    
    while i < len(line):
        matched = False
        for op in OPERATORS:
            if line.startswith(op, i):
                tokens.append(Token(TokenType.OPERATOR, op))
                i += len(op)
                matched = True
                break
        
        if matched:
            continue
        
        ch = line[i]
        
        if ch == "%":
            j = i + 1
            while j < len(line) and line[j] != "%":
                j += 1
            if j < len(line):
                tokens.append(Token(TokenType.VARREF, line[i:j+1]))
                i = j + 1
                continue
            else:
                tokens.append(Token(TokenType.TEXT, ch))
                i += 1
                continue
        
        if ch == "!":
            j = i + 1
            while j < len(line) and line[j] != "!":
                j += 1
            if j < len(line):
                tokens.append(Token(TokenType.DELAYED, line[i:j+1]))
                i = j + 1
                continue
            else:
                tokens.append(Token(TokenType.TEXT, ch))
                i += 1
                continue
        
        j = i + 1
        while j < len(line):
            if line[j] in "%!" or any(line.startswith(op, j) for op in OPERATORS):
                break
            j += 1
        
        tokens.append(Token(TokenType.TEXT, line[i:j]))
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
        
        tokens = tokenize_line(line)
        new_parts: List[str] = []
        
        for token in tokens:
            if token.type in (TokenType.LABEL, TokenType.VARREF, TokenType.DELAYED, TokenType.OPERATOR):
                new_parts.append(token.text)
                continue
            
            obfuscated = obfuscate_text(token.text, keys, values)
            new_parts.append(obfuscated)
        
        out_lines.append("".join(new_parts))
    
    logger.debug(f"Obfuscated {len(content.splitlines())} lines")
    return "\n".join(out_lines)


def obfuscate_text(text: str, keys: List[str], values: List[str]) -> str:
    buf: List[str] = []
    
    for ch in text:
        replaced = False
        for idx, val in enumerate(values):
            pos = val.find(ch)
            if pos != -1:
                buf.append(f"%{keys[idx]}:~{pos},1%")
                replaced = True
                break
        
        if not replaced:
            buf.append(ch)
    
    return "".join(buf)


def generate_set_lines(mapping: Dict[str, str]) -> List[str]:
    return [f'set "{k}={v}"' for k, v in mapping.items()]


def backup_if_exists(path: Path) -> None:
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, backup)
            logger.info(f"Created backup: {backup}")
        except IOError as e:
            logger.warning(f"Failed to create backup: {e}")


def process_file(path: Path, num_keys: int = NUM_KEYS, verbose: bool = False) -> Path:
    if not path.exists():
        raise InvalidFileError(f"File not found: {path}")
    
    if not path.is_file():
        raise InvalidFileError(f"Not a file: {path}")
    
    if path.suffix.lower() != ".bat":
        logger.warning(f"File extension is {path.suffix}, expected .bat")
    
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise InvalidFileError(f"Failed to read file: {e}")
    
    if verbose:
        logger.info(f"Processing: {path}")
        logger.info(f"File size: {len(raw)} bytes")
    
    cleaned = clean_comments(raw)
    mapping = generate_substrings(num_keys=num_keys)
    set_lines = generate_set_lines(mapping)
    body = obfuscate_with_substrings(cleaned, mapping)
    
    out = "@echo off\n" + "\n".join(set_lines) + "\n" + body + "\n"
    out_path = path.with_name(path.stem + "-obf" + path.suffix)
    
    backup_if_exists(out_path)
    
    try:
        out_path.write_text(out, encoding="utf-8")
        logger.info(f"Obfuscated output written to: {out_path}")
    except Exception as e:
        raise InvalidFileError(f"Failed to write output: {e}")
    
    if verbose:
        logger.info(f"Output size: {len(out)} bytes")
        logger.info(f"Compression ratio: {len(out)/len(raw):.2%}")
    
    return out_path


def cli() -> int:
    p = argparse.ArgumentParser(
        description="Batch (.bat) obfuscator (handles &&, ||, >, |, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s script.bat
  %(prog)s script.bat --keys 10
  %(prog)s script.bat -v
        """
    )
    p.add_argument("file", type=Path, help="Path to .bat file")
    p.add_argument(
        "--keys", 
        type=int, 
        default=NUM_KEYS, 
        help=f"Number of VAR keys to generate (default: {NUM_KEYS})"
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = p.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        out = process_file(args.file, num_keys=args.keys, verbose=args.verbose)
        print(f"Obfuscated batch written to: {out}")
        return 0
    except InvalidFileError as e:
        print(f"Invalid file: {e}", file=sys.stderr)
        return 1
    except MappingError as e:
        print(f"Mapping error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        logger.exception("Unexpected error:")
        return 2


if __name__ == "__main__":
    sys.exit(cli())
