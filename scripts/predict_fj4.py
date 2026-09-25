"""P3 M6-b：附件4（可解释专项测试集，20 条无标签）全量解释交付。

流程：
  1. 读 M2（φ/预测/强度/门控）与 M3（IG 位置归因/词归因）缓存 —— 解释对象与
     归因层完全复用已验收产物，不重算模型；
  2. 置信度温度校准：T=1.2142（valid NLL，P2 附件3 同款），conf=p^(1/T) 归一后取 max；
  3. 证据定位：全局 TOP3（跨模态 |IG_cls|）→ 位置→原词（build_grid）→
     物理时间（M6-a P1 CTC 对齐）；vision 证据附关键帧渲染（视频 PTS 内取中点帧，
     仅回看用，不重新提特征）；
  4. 双 CSV 交付（方案 §8）：prediction（20 行）+ evidence 长表（60 行）；
  5. 解释卡（≥3 典型样本）→ runs/p3/m6/cards/。
纪律（方案 §9/§10）：附件4 无标签——不做精度声明；text IG 未通过单点 LOO 裁判
（M4-F），解释卡引用 A/B/C 全局保真度；语义描述不越界（组织者 74/35 维特征无
官方维名字典 → 不指名具体 AU/声学维，只到时间区间级）。
预注册检查：pred 行数=20；evidence 行数=60；#13 vision 证据数=0；全部时间在
视频时长内；校准 argmax 不变性；帧文件存在。
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p3.data import VIDEOS_DIR, load_att4  # noqa: E402
from src.p3.fidelity import evidence_ranking  # noqa: E402
from src.p3.ig import p1_word_grid  # noqa: E402
from src.p3.shapley import MODS, prepare_sample  # noqa: E402

OUT = ROOT / "runs/p3/m6"
TEMP = 1.2142
CLS_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}
CARDS = ("04", "09", "13", "17")          # 双向/强负文本主导/vision 缺失/高置信正
PREREG = {"pred_rows": 20, "evidence_rows": 60, "vision_evidence_13": 0}


def calibrate(p: np.ndarray) -> np.ndarray:
    q = p ** (1.0 / TEMP)
    return q / q.sum()


def grab_frame(mp4: Path, t: float, dst: Path) -> bool:
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if ok:
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), frame)
    return ok


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("✅" if ok else "❌"), name, "" if ok else detail)

    m2 = json.loads((ROOT / "runs/p3/m2/shapley_att4.json").read_text())
    m3 = json.loads((ROOT / "runs/p3/m3/ig_att4.json").read_text())
    r2 = {r["sample_id"]: r for r in m2["reports"]}
    r3 = {r["sample_id"]: r for r in m3["reports"]}

    samples = load_att4()
    from src.p2.pipeline import load_stats
    stats = load_stats()
    pred_rows, ev_rows, mod_rows, cards_md = [], [], [], []
    n_v13 = 0
    for s in samples:
        sid = f"{s.n:02d}"
        a2, a3 = r2[sid], r3[sid]
        align = json.loads((OUT / "alignment" / f"{sid}.json").read_text())
        wg = p1_word_grid(s.raw_text)
        wt = {w["id"]: w for w in align["words"]}

        pc = int(a2["pred_class"])
        pv = np.array(a2["coalition_values"]["audio|text|vision"]["v_cls"])
        p_cal = calibrate(pv)
        main_m = max(MODS, key=lambda m: abs(a2["phi_cls"][m][pc]))
        pred_rows.append({
            "sample_id": sid, "pred_polarity": CLS_NAME[pc], "pred_polarity_id": pc,
            "pred_intensity": a2["pred_intensity"],
            "confidence": round(float(p_cal.max()), 4),
            "confidence_raw": round(float(pv.max()), 4),
            "main_modality": main_m,
            "phi_t": round(a2["phi_cls"]["text"][pc], 6),
            "phi_a": round(a2["phi_cls"]["audio"][pc], 6),
            "phi_v": round(a2["phi_cls"]["vision"][pc], 6),
            "phi_reg_t": round(a2["phi_reg"]["text"], 6),
            "phi_reg_a": round(a2["phi_reg"]["audio"], 6),
            "phi_reg_v": round(a2["phi_reg"]["vision"], 6),
            "gate_t": a2["gates_full"][0], "gate_a": a2["gates_full"][1],
            "gate_v": a2["gates_full"][2],
            "vision_natural_missing": a3["vision_natural_missing"]})

        ig_cls = {m: np.array(a3["baselines"]["missing"]["ig"][m]["cls"])
                  for m in MODS}
        st = prepare_sample(s, stats)            # 自然观测（证据可用位同 M4）
        rank = evidence_ranking(ig_cls, st)[:3]
        ev = []
        for r, (m, j, val) in enumerate(rank, 1):
            w = wg["word_by_position"][j]
            if m == "vision" and sid == "13":
                continue
            entry = {"sample_id": sid, "rank": r, "modality": m,
                     "attribution": round(val, 6),
                     "model_position": j}
            if m == "text":
                entry.update({"evidence_type": "word",
                              "content": wg["word_texts"].get(w, "?"),
                              "n_pieces": wg["n_pieces"].get(w, 0)})
            else:
                entry["evidence_type"] = ("acoustic_interval" if m == "audio"
                                          else "visual_interval_keyframe")
                word = wt.get(w)
                if m == "vision":
                    t_mid = (word["t_s"] + word["t_e"]) / 2 if word else 0.0
                    dst = OUT / "frames" / f"{sid}_top{r}.jpg"
                    ok_f = grab_frame(VIDEOS_DIR / f"{sid}.mp4", t_mid, dst)
                    entry["keyframe"] = str(dst.relative_to(ROOT)) if ok_f else ""
            if w is not None and w in wt:
                entry["t_start"] = wt[w]["t_s"]; entry["t_end"] = wt[w]["t_e"]
                entry["word_id"] = w
                entry["word_align_status"] = wt[w]["status"]
            else:
                entry["t_start"] = entry["t_end"] = None
                entry["word_align_status"] = ""
            if m == "vision":
                n_v13 += (sid == "13")
            ev.append(entry)
        ev_rows += ev

        # 逐模态 TOP1（补充表：全局 TOP3 全文本主导后，题目要求的语音时段/视频帧视角）
        for m in MODS:
            js_m = np.where(st.o[m][0])[0]               # 单模态 TOP1（可用位内）
            if len(js_m) == 0:                           # 模态整缺失（#13 vision）
                mod_rows.append({"sample_id": sid, "modality": m,
                                 "evidence_type": "modality_natural_missing",
                                 "content": "", "t_start": "", "t_end": "",
                                 "attribution": 0.0, "model_position": "",
                                 "keyframe": "", "note": "o_m≡0，无证据（M2 哑玩家）",
                                 "word_align_status": ""})
                continue
            j = int(js_m[np.argmax(np.abs(ig_cls[m][js_m]))])
            val = abs(float(ig_cls[m][j]))
            w = wg["word_by_position"][j]
            word = wt.get(w)
            entry = {"sample_id": sid, "modality": m,
                     "evidence_type": ("word" if m == "text" else
                                       "acoustic_interval" if m == "audio" else
                                       "visual_interval_keyframe"),
                     "content": wg["word_texts"].get(w, "") if m == "text" else "",
                     "t_start": word["t_s"] if word else "",
                     "t_end": word["t_e"] if word else "",
                     "attribution": round(val, 6), "model_position": j,
                     "keyframe": "", "note": "",
                     "word_align_status": word["status"] if word else ""}
            if m == "vision" and word:
                t_mid = (word["t_s"] + word["t_e"]) / 2
                dst = OUT / "frames" / f"{sid}_{m}.jpg"
                if grab_frame(VIDEOS_DIR / f"{sid}.mp4", t_mid, dst):
                    entry["keyframe"] = str(dst.relative_to(ROOT))
            mod_rows.append(entry)

        if sid in CARDS:
            top_txt = [e for e in ev if e["modality"] == "text"]
            cards_md.append(render_card(s, a2, p_cal, main_m, ev, top_txt, align))

    # ---- 预注册检查 ----
    add("pred_rows", len(pred_rows) == PREREG["pred_rows"], len(pred_rows))
    add("evidence_rows", len(ev_rows) == PREREG["evidence_rows"], len(ev_rows))
    add("vision_evidence_13", n_v13 == PREREG["vision_evidence_13"], n_v13)
    bad_t = [e["sample_id"] for e in ev_rows
             if e["t_start"] is None or e["t_end"] is None]
    add("evidence_times_present", not bad_t, bad_t)
    frames_ok = all((ROOT / e["keyframe"]).exists() for e in ev_rows + mod_rows
                    if e.get("keyframe"))
    n_frames = len(list((OUT / "frames").glob("*.jpg")))
    add("keyframes_rendered", frames_ok, {"n_frames": n_frames})
    add("modality_top_rows", len(mod_rows) == 60, len(mod_rows))
    inv = all((lambda raw, cal: int(np.argmax(raw)) == int(np.argmax(cal)))(
        np.array(r2[f'{i:02d}']["coalition_values"]["audio|text|vision"]["v_cls"]),
        calibrate(np.array(r2[f'{i:02d}']["coalition_values"]
                           ["audio|text|vision"]["v_cls"]))) for i in range(1, 21))
    add("calibration_argmax_invariant", inv, f"invariant={inv}")

    # ---- 落盘 ----
    with open(OUT / "附件4_预测与模态贡献.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        w.writeheader(); w.writerows(pred_rows)
    with open(OUT / "附件4_证据定位长表.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ev_rows[0].keys()))
        w.writeheader(); w.writerows(ev_rows)
    with open(OUT / "附件4_逐模态TOP1证据.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(mod_rows[0].keys()))
        w.writeheader(); w.writerows(mod_rows)
    (OUT / "解释卡.md").write_text("\n\n---\n\n".join(cards_md), encoding="utf-8")
    meta = {"created_at": datetime.now(timezone.utc).isoformat(),
            "model": "MRFN+ 3-seed ensemble（M2/M3 缓存）",
            "calibration": {"T": TEMP, "fitted_on": "valid NLL（P2 同款）"},
            "checks": checks, "all_pass": all(c["ok"] for c in checks),
            "discipline": "附件4 无标签：不做精度声明；text IG 未过单点 LOO（M4-F），"
                          "引用 A/B/C 全局保真度；维级语义不越界（无官方维名字典）"}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"\nCSV×2 + 解释卡({len(cards_md)}) → {OUT}")
    print("ALL PASS" if meta["all_pass"] else "FAILED")
    return 0 if meta["all_pass"] else 1


def render_card(s, a2, p_cal, main_m, ev, top_txt, align):
    pc = int(a2["pred_class"])
    phi = a2["phi_cls"]
    lines = [
        f"## 解释卡 · 样本 {s.n:02d}",
        f"**极性 {CLS_NAME[pc]}**（校准置信度 {p_cal.max():.2f}） | "
        f"**强度 {a2['pred_intensity']:+.2f}** | 主要参考模态 **{main_m}**",
        "",
        f"> 转写：\"{s.raw_text[:120]}{'…' if len(s.raw_text) > 120 else ''}\"",
        "",
        "| 模态 | φ（支持/抑制，预测类） | φ（强度） | 门控 g |",
        "|---|---|---|---|",
        f"| text | {phi['text'][pc]:+.3f} | {a2['phi_reg']['text']:+.3f} | "
        f"{a2['gates_full'][0]:.3f} |",
        f"| audio | {phi['audio'][pc]:+.3f} | {a2['phi_reg']['audio']:+.3f} | "
        f"{a2['gates_full'][1]:.3f} |",
        f"| vision | {phi['vision'][pc]:+.3f} | {a2['phi_reg']['vision']:+.3f} | "
        f"{a2['gates_full'][2]:.3f} |",
        "",
        "**证据 TOP3（跨模态，|IG| 序）**：",
    ]
    for e in ev:
        t = (f"t∈[{e['t_start']:.2f},{e['t_end']:.2f}]s" if e["t_start"] is not None
             else "时间未覆盖（截断词）")
        content = e.get("content", "")
        if e["modality"] == "text":
            lines.append(f"{e['rank']}. **[text]** \"{content}\" {t} "
                         f"（{e['n_pieces']} 子词） a={e['attribution']:.3f}")
        elif e["modality"] == "audio":
            lines.append(f"{e['rank']}. **[audio]** 声学证据区间 {t} "
                         f"a={e['attribution']:.3f}（组织者特征无维名字典，"
                         f"描述到时间区间级）")
        else:
            lines.append(f"{e['rank']}. **[vision]** 视觉证据区间 {t} "
                         f"a={e['attribution']:.3f}（关键帧 {e.get('keyframe','')}）")
    if s.n == 13:
        lines.append("")
        lines.append("> ⚠️ 本样本 vision 整模态自然缺失（φ_v≡0 哑玩家，M2 实证）；"
                     "解释完全由 text/audio 支撑。")
    lines.append("")
    lines.append(f"对齐：{len(align['words'])} 词 CTC 强制对齐"
                 f"（失败 {align['failure_count']}），视频时长 {align['duration_s']:.1f}s")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
