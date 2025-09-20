# convert_segments_to_boxes.py
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Decentra/datasets")  # adjust if needed
splits = ["train", "valid", "test"]

def line_to_bbox(line: str):
    parts = line.strip().split()
    if len(parts) <= 5:
        return line  # already bbox or empty
    cls = parts[0]
    nums = list(map(float, parts[1:]))
    xs = nums[0::2]; ys = nums[1::2]
    xmin, xmax = max(0.0, min(xs)), min(1.0, max(xs))
    ymin, ymax = max(0.0, min(ys)), min(1.0, max(ys))
    cx = (xmin + xmax)/2.0
    cy = (ymin + ymax)/2.0
    w  = max(1e-6, xmax - xmin)
    h  = max(1e-6, ymax - ymin)
    return f"{int(float(cls))} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"

for sp in splits:
    lbl_dir = ROOT/sp/"labels"
    if not lbl_dir.exists(): continue
    for txt in lbl_dir.rglob("*.txt"):
        lines = txt.read_text(encoding="utf-8").splitlines()
        if not lines: continue
        need = any(len(l.split()) > 5 for l in lines)
        if not need: continue
        new_lines = [line_to_bbox(l) for l in lines if l.strip()]
        txt.write_text("".join(new_lines), encoding="utf-8")
        print("Converted:", txt)
