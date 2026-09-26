"""BERT 50 格网格与 WordPiece→发音词映射（问题一 §2.1/§2.3/§4）。

网格严格使用 ``[CLS] + 前 48 个非特殊 token + [SEP]``。tokenizer 的
``word_ids()`` 会把缩写和标点拆成伪词，因此这里只使用 ``offset_mapping``
记录原文字符区间；WordPiece 到发音词的归属由 :mod:`normalize` 的字符消费式
映射决定。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .normalize import build_wp_word_map, normalize_transcript

GRID_SIZE = 50
MAX_CONTENT_PIECES = GRID_SIZE - 2
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "cache" / "hf" / "bert-base-uncased"


@lru_cache(maxsize=None)
def load_tokenizer(
    model_path: str | Path = DEFAULT_TOKENIZER_PATH,
) -> PreTrainedTokenizerBase:
    """从冻结的本地目录加载 fast bert-base-uncased tokenizer。"""
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    if not tokenizer.is_fast:
        raise RuntimeError("P1 grid requires a fast tokenizer for offset_mapping")
    required_ids = {
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    missing = [name for name, value in required_ids.items() if value is None]
    if missing:
        raise RuntimeError(f"tokenizer is missing required special ids: {missing}")
    return tokenizer


def _as_list(value: Any) -> list:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _word_char_span(entry: dict, offsets: dict[int, tuple[int, int]]) -> tuple[int, int]:
    spans = [offsets[position] for position in entry["wp_ids"] if position in offsets]
    if not spans:
        return -1, -1
    return min(start for start, _ in spans), max(end for _, end in spans)


def build_grid(
    raw_text: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """把一条原始转写变为固定 50 格网格及可回溯映射表。

    ``alpha`` 在每个发音词实际保留的网格位置间均分；若一个词跨越第 48
    个内容片边界，只对保留片重新归一化。特殊位和 padding 的 alpha 为零。
    """
    if not isinstance(raw_text, str):
        raise TypeError(f"raw_text must be str, got {type(raw_text).__name__}")
    if not raw_text.strip():
        raise ValueError("raw_text must contain at least one non-space character")

    tokenizer = tokenizer or load_tokenizer()
    encoded = tokenizer(
        raw_text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_offsets_mapping=True,
    )
    full_ids = [int(value) for value in _as_list(encoded["input_ids"])]
    full_offsets = [tuple(map(int, pair)) for pair in _as_list(encoded["offset_mapping"])]
    if len(full_ids) != len(full_offsets):
        raise RuntimeError("tokenizer returned inconsistent ids and offset_mapping")

    full_tokens = tokenizer.convert_ids_to_tokens(full_ids)
    words_text = normalize_transcript(raw_text).split()
    full_entries, anomalies, unmapped_words = build_wp_word_map(
        [tokenizer.cls_token, *full_tokens, tokenizer.sep_token], words_text
    )
    if unmapped_words:
        anomalies.append(f"{unmapped_words} normalized words were not mapped")

    retained_ids = full_ids[:MAX_CONTENT_PIECES]
    retained_count = len(retained_ids)
    sep_position = retained_count + 1
    padding_count = GRID_SIZE - sep_position - 1
    input_ids = np.asarray(
        [tokenizer.cls_token_id, *retained_ids, tokenizer.sep_token_id]
        + [tokenizer.pad_token_id] * padding_count,
        dtype=np.int64,
    )
    attention_mask = np.zeros(GRID_SIZE, dtype=np.int64)
    attention_mask[: sep_position + 1] = 1
    token_type_ids = np.zeros(GRID_SIZE, dtype=np.int64)
    content_mask = np.zeros(GRID_SIZE, dtype=np.uint8)
    content_mask[1:sep_position] = 1
    special_mask = np.zeros(GRID_SIZE, dtype=np.uint8)
    special_mask[[0, sep_position]] = 1
    padding_mask = (attention_mask == 0).astype(np.uint8)
    alpha = np.zeros(GRID_SIZE, dtype=np.float32)

    offsets_by_position = {
        position: full_offsets[position - 1]
        for position in range(1, len(full_offsets) + 1)
    }
    position_to_word: dict[int, tuple[dict, str]] = {}
    full_word_map = []
    for entry in full_entries:
        attach_by_position = {position: flag for position, flag in entry["inherit_flags"]}
        mapped_positions = sorted(set(entry["wp_ids"]) | set(attach_by_position))
        retained_positions = [
            position for position in mapped_positions if 1 <= position <= retained_count
        ]
        retained_lexical = [
            position for position in entry["wp_ids"] if position <= retained_count
        ]
        word_start, word_end = _word_char_span(entry, offsets_by_position)
        trunc_flag = any(position > retained_count for position in mapped_positions)
        dropped = not retained_lexical
        full_word_map.append({
            "word_id": entry["word_id"],
            "word_text": entry["text"],
            "char_start": word_start,
            "char_end": word_end,
            "wp_positions": list(entry["wp_ids"]),
            "attach_positions": list(entry["inherit_flags"]),
            "retained_positions": retained_positions,
            "trunc_flag": trunc_flag,
            "dropped": dropped,
        })
        if not retained_positions:
            continue
        weight = np.float32(1.0 / len(retained_positions))
        for position in retained_positions:
            if position in position_to_word:
                anomalies.append(f"wp{position} was assigned to multiple words")
                continue
            attach_type = attach_by_position.get(position, "word")
            position_to_word[position] = (entry, attach_type)
            alpha[position] = weight

    grid_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    wp_word_map = []
    for position in range(GRID_SIZE):
        if padding_mask[position]:
            kind = "padding"
        elif special_mask[position]:
            kind = "special"
        else:
            kind = "content"

        mapped = position_to_word.get(position)
        if mapped is None:
            word_id = None
            word_text = None
            wp_count = 0
            attach_type = kind
            trunc_flag = False
        else:
            entry, attach_type = mapped
            word_id = entry["word_id"]
            word_text = entry["text"]
            retained_positions = full_word_map[word_id]["retained_positions"]
            wp_count = len(retained_positions)
            trunc_flag = full_word_map[word_id]["trunc_flag"]

        if kind == "content":
            char_start, char_end = offsets_by_position[position]
            if mapped is None:
                anomalies.append(f"wp{position} '{grid_tokens[position]}' has no word mapping")
        else:
            char_start, char_end = -1, -1
        wp_word_map.append({
            "position": position,
            "token_id": int(input_ids[position]),
            "token": grid_tokens[position],
            "char_start": char_start,
            "char_end": char_end,
            "word_id": word_id,
            "word_text": word_text,
            "wp_count_in_word": wp_count,
            "alpha": float(alpha[position]),
            "attach_type": attach_type,
            "trunc_flag": trunc_flag,
        })

    partition = content_mask.astype(np.int8) + special_mask + padding_mask
    if not np.all(partition == 1):
        raise RuntimeError("content/special/padding masks do not form a partition")

    retained_word_count = sum(not item["dropped"] for item in full_word_map)
    return {
        "raw_text": raw_text,
        "normalized_text": " ".join(words_text),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "content_mask": content_mask,
        "special_mask": special_mask,
        "padding_mask": padding_mask,
        "alpha": alpha,
        "wp_word_map": wp_word_map,
        "full_word_map": full_word_map,
        "sep_position": sep_position,
        "full_wp_count": len(full_ids),
        "retained_wp_count": retained_count,
        "truncated": len(full_ids) > MAX_CONTENT_PIECES,
        "retained_word_count": retained_word_count,
        "dropped_word_count": len(full_word_map) - retained_word_count,
        "anomalies": anomalies,
    }
