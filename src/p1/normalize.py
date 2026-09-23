"""转写规范化与 wp–word 映射（问题一方案 §2.3 / v2 修订稿）。

对齐词序列 = 规范化转写的空白词；wp 网格 = bert-base-uncased 对**原始转写**的
wordpiece 序列（与附件2 同构，标点位置存在）。映射按"字符消费"对齐：

- 词片段（含 ## 续片）逐片累积，累积串 == 词文本 时收词；
- 词中标点（缩字撇号 they've 的 '）：buf 非空 → 消费进当前词，标 mid-*；
- 独立标点（buf 为空）：不消费——收尾/两可→挂前词，起始→暂存并入后词；
- [CLS]/[SEP]/[PAD]/[UNK] 跳过（UNK 极罕见，计入 anomalies 人工核查）。
"""
import re

_TAIL = set(",.:;!?')%]>…。，；：！？'")
_HEAD = set('(“{"[$#«¿¡')
_AMBI = set('"-–—')
_SPECIAL = {"[CLS]", "[SEP]", "[PAD]"}


def _canonical(text: str) -> str:
    """弯引号/长破折号 → ASCII（token 侧与转写侧共用，防两侧失配）。"""
    return (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
    )


def normalize_transcript(text: str) -> str:
    """对齐用转写规范化：小写、弯引号→ASCII、去标点（保留缩字撇号）、压空白。"""
    t = re.sub(r"[^\w\s']+", " ", _canonical(text).lower())
    # 词缘撇号剥离（引号用法），保留词内缩字撇号（they've）
    words = [w.strip("'") for w in t.split()]
    return re.sub(r"\s+", " ", " ".join(w for w in words if w)).strip()


def punct_class(token: str) -> str | None:
    """继承类别：tail / head / ambi / punct（未登记纯标点）/ None（含词字符）。"""
    frag = token[2:] if token.startswith("##") else token
    if re.search(r"[\w']", frag) and frag != "'":
        return None
    if frag in _TAIL:
        return "tail"
    if frag in _HEAD:
        return "head"
    if frag in _AMBI:
        return "ambi"
    return "punct"  # 未登记标点（如 ^）：位置规则挂靠，见 build_wp_word_map


def build_wp_word_map(
    tokens: list[str], words_text: list[str]
) -> tuple[list[dict], list[str], int]:
    """构建 wp→word 映射表（头部截断语义，§4）。

    返回 (entries, anomalies, n_dropped)：
      entries: {word_id, text, wp_ids, inherit_flags, trunc}
        trunc=True 表示 50 网格只保留了该词的前缀片段（池化向量只复制到保留片）；
      n_dropped: 整词越界、未入网格的词数（预期行为，非异常）；
      anomalies 非空 → 该样本进人工核查清单。
    """
    entries: list[dict] = []
    pending_head: list[tuple[int, str]] = []
    anomalies: list[str] = []
    wi, buf = 0, ""
    cur_wps: list[int] = []
    cur_punct: list[tuple[int, str]] = []
    trunc = False

    def close(flag_trunc: bool = False) -> None:
        nonlocal wi, buf, cur_wps, cur_punct, trunc
        entries.append({
            "word_id": len(entries),
            "text": words_text[wi],
            "wp_ids": list(cur_wps),
            "inherit_flags": list(cur_punct) + list(pending_head),
            "trunc": flag_trunc,
        })
        pending_head.clear()
        wi += 1
        buf, cur_wps, cur_punct = "", [], []
        trunc = False

    for tid, tok in enumerate(tokens):
        if tok in _SPECIAL:
            continue
        if tok == "[UNK]":
            anomalies.append(f"wp{tid} [UNK]")
            continue
        frag = _canonical(tok[2:] if tok.startswith("##") else tok)
        cls = punct_class(_canonical(tok))

        if buf == "" and cls is not None:  # 独立标点，不消费
            if cls in ("tail", "ambi") and entries:
                entries[-1]["inherit_flags"].append((tid, cls))
            else:  # head / 未登记标点在词首 → 挂后词；句首尾标 → 亦归首词
                pending_head.append((tid, cls))
            continue

        if wi >= len(words_text):
            anomalies.append(f"wp{tid} '{tok}' 越过末词")
            continue
        cand = buf + frag
        if cls is None and not words_text[wi].startswith(cand):
            anomalies.append(f"wp{tid} '{tok}' 与词 '{words_text[wi]}' 失配(buf='{buf}')")
            continue
        buf = cand
        cur_wps.append(tid)
        if cls is not None:
            cur_punct.append((tid, "mid-" + cls))
        if buf == words_text[wi]:
            close()

    if buf:  # 词被 50 网格截去尾部：保留已入网格片段，标 trunc
        trunc = True
        close(flag_trunc=True)
    n_dropped = len(words_text) - wi
    return entries, anomalies, n_dropped
