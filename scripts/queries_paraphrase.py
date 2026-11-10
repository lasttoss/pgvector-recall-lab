#!/usr/bin/env python3
"""Paraphrase queries: the words a customer uses, not the words in the catalogue.

This is the set that separates lexical from vector retrieval. Nothing in the
query appears verbatim in the product name, so BM25 has nothing to match.
"""
import json, random

SYN = {
    "den led am tran": ["đèn gắn trần ốp trần", "đèn chiếu điểm trên trần", "downlight âm trần"],
    "den led roi": ["đèn chiếu tường điểm nhấn", "đèn rọi tranh", "spotlight trần"],
    "den tha tran": ["đèn treo giữa phòng khách", "đèn chùm thả xuống bàn ăn"],
    "den thanh led": ["đèn tuýp dài gắn trần", "đèn hộp mảnh cho hành lang"],
    "den ngoai troi": ["đèn sân vườn chống mưa", "đèn trụ cổng"],
    "voi lavabo": ["vòi nước cho chậu rửa mặt", "vòi chậu rửa"],
    "bon cau": ["bệ xí vệ sinh", "toilet cho phòng tắm"],
    "sen tam": ["vòi hoa sen tắm đứng", "bộ sen tắm nóng lạnh"],
    "chau lavabo": ["chậu rửa mặt đặt trên bàn đá", "lavabo treo tường"],
    "o cam dien": ["ổ điện cắm phích", "đế cắm điện âm tường"],
    "cong tac dien": ["nút bấm bật tắt đèn", "công tắc chiếu sáng"],
    "da granite": ["mặt bàn đá tự nhiên", "đá ốp bếp"],
    "go cong nghiep": ["tấm gỗ ép làm tủ", "gỗ ván cho nội thất"],
}
CCT_WORDS = {"3000K": "ánh sáng vàng ấm", "4000K": "ánh sáng trung tính", "6500K": "ánh sáng trắng lạnh"}

def main(n=40, seed=23):
    rows = json.load(open("out/catalogue.json"))
    rnd = random.Random(seed)
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["cat"], []).append(r)
    picks = rnd.sample(rows, n)
    out = []
    for r in picks:
        base = rnd.choice(SYN[r["cat"]])
        q = base
        extra = None
        if "den" in r["cat"]:
            cct = [c for c in CCT_WORDS if c in r["desc"]]
            if cct:
                q = f"{base} {CCT_WORDS[cct[0]]}"
                extra = cct[0]
        # Tighten the gold set: same category, same colour and (for lighting) the
        # same colour temperature. A loose gold set of 50 items makes recall@10
        # arithmetically meaningless.
        gold = [x["id"] for x in by_cat[r["cat"]]
                if x["color"] == r["color"] and (extra is None or extra in x["desc"])]
        if len(gold) < 2:
            gold = [x["id"] for x in by_cat[r["cat"]] if extra is None or extra in x["desc"]][:8]
        out.append({"q": q, "gold": gold})
    json.dump(out, open("out/queries_paraphrase.json", "w"), ensure_ascii=False, indent=0)
    print(f"wrote {len(out)} paraphrase queries, avg gold {sum(len(q['gold']) for q in out)/len(out):.1f}")

if __name__ == "__main__":
    main()
