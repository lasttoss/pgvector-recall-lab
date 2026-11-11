#!/usr/bin/env python3
"""A query set large enough to separate models (250 queries, four slices).

The 40-query set could not tell any two models apart: the 95% CI of the difference
was ±0.2, wider than every effect measured. This generator builds 250 queries whose
gold sets are computed from catalogue attributes only — never from a model — and
rejects any query whose gold set is too small (noisy) or too large (hit@10 becomes
trivial). Four slices, same attributes, different wording:
  plain   base phrase + attributes
  nodauc  the same with Vietnamese diacritics stripped (how people type on phones)
  verbose a full sentence with filler words that carry no signal
  brand   brand name is the discriminator, colour is left out
"""
import json, random, unicodedata

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
FILLER = ["cần mua gấp", "cho công trình", "giá hợp lý", "loại tốt", "đang cần", "để lắp đặt"]


def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def main(target=250, seed=11):
    rows = json.load(open("out/catalogue.json"))
    rnd = random.Random(seed)
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["cat"], []).append(r)

    def gold_for(cat, color=None, cct=None, brand=None):
        out = []
        for x in by_cat[cat]:
            if color is not None and x["color"] != color:
                continue
            if brand is not None and x["brand"] != brand:
                continue
            if cct is not None and cct not in x["desc"]:
                continue
            out.append(x["id"])
        return out

    slices = ["plain", "nodauc", "verbose", "brand"]
    queries, seen = [], set()
    guard = 0
    while len(queries) < target and guard < target * 400:
        guard += 1
        sl = slices[len(queries) % len(slices)]
        r = rnd.choice(rows)
        base = rnd.choice(SYN[r["cat"]])
        cct = next((c for c in CCT_WORDS if c in r["desc"]), None) if "den" in r["cat"] else None
        if sl == "brand":
            q = f"{r['brand']} {base}"
            if cct:
                q += " " + CCT_WORDS[cct]
                gold = gold_for(r["cat"], cct=cct, brand=r["brand"])
            else:
                gold = gold_for(r["cat"], brand=r["brand"])
        else:
            q = f"{base} {r['color']}"
            if cct:
                q += " " + CCT_WORDS[cct]
            gold = gold_for(r["cat"], color=r["color"], cct=cct)
            if sl == "verbose":
                q = f"{rnd.choice(FILLER)}, tôi đang tìm {q}"
            elif sl == "nodauc":
                q = strip_accents(q)
        if not (3 <= len(gold) <= 25):
            continue
        key = (q, tuple(sorted(gold)))
        if key in seen:
            continue
        seen.add(key)
        queries.append({"q": q, "gold": gold, "slice": sl, "gold_size": len(gold)})

    rnd.shuffle(queries)
    json.dump(queries, open("out/queries_power.json", "w"), ensure_ascii=False, indent=0)
    by_slice = {}
    for q in queries:
        by_slice.setdefault(q["slice"], []).append(q["gold_size"])
    print(f"wrote {len(queries)} queries")
    for k, v in sorted(by_slice.items()):
        print(f"  {k:8} n={len(v):3}  gold avg={sum(v)/len(v):5.1f} min={min(v)} max={max(v)}")
    for q in queries[:4]:
        print(f"  [{q['slice']:7}] {q['q']}  (gold {q['gold_size']})")


if __name__ == "__main__":
    main()
