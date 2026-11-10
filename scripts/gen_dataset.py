#!/usr/bin/env python3
"""Generate a deterministic Vietnamese building-materials catalogue.

Synthetic on purpose: the point is to measure the *pipeline* (lexical vs vector
vs hybrid), not to claim anything about real-world search quality.
"""
import json, random

BRANDS = ["Rang Dong", "Dien Quang", "Philips", "Panasonic", "Kohn", "Inax",
          "Toto", "Caesar", "Hafele", "Xingfa", "Vicostone", "An Cuong"]
CATS = [
    ("den led am tran", "đèn LED âm trần", ["7W", "9W", "12W", "18W"], ["90mm", "110mm"]),
    ("den led roi", "đèn LED rọi", ["3W", "5W", "7W"], ["35mm", "50mm"]),
    ("den tha tran", "đèn thả trần", ["12W", "20W", "30W"], ["300mm", "450mm"]),
    ("den thanh led", "đèn thanh LED", ["10W", "20W", "36W"], ["600mm", "1200mm"]),
    ("den ngoai troi", "đèn ngoài trời", ["6W", "12W", "24W"], ["trụ 500mm", "trụ 800mm"]),
    ("voi lavabo", "vòi lavabo", ["nóng lạnh", "lạnh"], ["âm tường", "nổi"]),
    ("bon cau", "bồn cầu", ["1 khối", "2 khối"], ["bàn cầu", "êm"]),
    ("sen tam", "sen tắm", ["nóng lạnh", "thường"], ["dây sen", "âm tường"]),
    ("chau lavabo", "chậu lavabo", ["đặt bàn", "âm bàn"], ["lòng chảo", "vuông"]),
    ("o cam dien", "ổ cắm điện", ["3 chấu", "2 chấu"], ["âm tường", "nổi"]),
    ("cong tac dien", "công tắc điện", ["1 chiều", "2 chiều"], ["âm tường", "nổi"]),
    ("da granite", "đá granite", ["tự nhiên", "nhân tạo"], ["2cm", "3cm"]),
    ("go cong nghiep", "gỗ công nghiệp", ["MDF", "HDF"], ["chống ẩm", "thường"]),
]
CCT = ["3000K", "4000K", "6500K"]
COLORS = ["trắng", "đen", "xám", "vàng đồng"]
WARRANTY = ["2 năm", "3 năm", "5 năm"]

def main(n=900, seed=7):
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        key, vi, variants, sizes = rnd.choice(CATS)
        brand = rnd.choice(BRANDS)
        variant = rnd.choice(variants)
        size = rnd.choice(sizes)
        cct = rnd.choice(CCT) if "den" in key else ""
        color = rnd.choice(COLORS)
        warranty = rnd.choice(WARRANTY)
        price = rnd.choice([180_000, 250_000, 320_000, 450_000, 690_000, 1_250_000])
        name = " ".join(x for x in [vi, brand, variant, cct, size, color] if x)
        desc = (f"{vi} {brand}, {variant}, kích thước {size}, màu {color}"
                + (f", nhiệt độ màu {cct}" if cct else "")
                + f". Bảo hành {warranty}, phù hợp công trình dân dụng và nội thất cao cấp.")
        rows.append({"id": i + 1, "name": name, "name_ascii": None, "desc": desc,
                     "cat": key, "brand": brand, "variant": variant, "size": size,
                     "color": color, "price": price})
    json.dump(rows, open("out/catalogue.json", "w"), ensure_ascii=False, indent=0)
    print(f"wrote {len(rows)} items")

if __name__ == "__main__":
    main()
