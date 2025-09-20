"""Download and merge multiple Roboflow datasets to a single YOLO-format dataset.
Usage:
export ROBOFLOW_API_KEY=...
python data/download_roboflow.py --projects "user/proj1:3,org/proj2:5" \
--target ./datasets/damage_merged --format yolo
"""
import os, shutil, argparse, re
from pathlib import Path
from roboflow import Roboflow

import os, json, random
import numpy as np, torch
from pathlib import Path


def set_seed(seed: int = 42):
  random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
  if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
  Path(p).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
  with open(path, 'w') as f: json.dump(obj, f, indent=2)


def load_json(path):
  with open(path, 'r') as f: return json.load(f)


SUPPORTED_FORMATS = {"yolov8", "coco"}


def parse_projects(s: str):
  # input like: "user/proj1:3,org/proj2:5"
  items = [x.strip() for x in s.split(',') if x.strip()]
  out = []
  for it in items:
    m = re.match(r"([^/]+)/([^:]+):(\d+)", it)
    if not m:
      raise ValueError(f"Bad project spec: {it}. Use owner/project:version")
    out.append({"owner": m.group(1), "project": m.group(2), "version": int(m.group(3))})
    return out


def copytree(src, dst):
  for root, dirs, files in os.walk(src):
    rel = os.path.relpath(root, src)
    odir = os.path.join(dst, rel)
    os.makedirs(odir, exist_ok=True)
    for f in files:
      s = os.path.join(root, f); d = os.path.join(odir, f)
      if not os.path.exists(d): shutil.copy2(s, d)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--projects', required=True, help='Comma list owner/project:version')
  ap.add_argument('--target', required=True)
  ap.add_argument('--format', default='yolo', choices=SUPPORTED_FORMATS)
  args = ap.parse_args()


  api_key = os.environ.get('ROBOFLOW_API_KEY')
  if not api_key:
    raise SystemExit('Set ROBOFLOW_API_KEY env var')


  rf = Roboflow(api_key=api_key)
  target = Path(args.target)
  images = target / 'images'
  labels = target / 'labels'
  for p in [images, labels]: ensure_dir(p)


  projects = parse_projects(args.projects)
  print('Will download:', projects)


  for spec in projects:
    ws = rf.workspace(spec['owner'])
    proj = ws.project(spec['project'])
    ver = proj.version(spec['version'])
    print(f"Downloading {spec} ...")
    ds_dir = ver.download(args.format).location
    # Expect YOLO structure: {train,val,test}/{images,labels}
    for split in ['train', 'valid', 'val', 'test']:
      sp = Path(ds_dir)/split
    if not sp.exists(): continue
    for sub in ['images','labels']:
      sdir = sp/sub
    if sdir.exists():
      copytree(str(sdir), str(target/sub))
  print('✅ Merge complete at', target)


if __name__ == '__main__':
  main()