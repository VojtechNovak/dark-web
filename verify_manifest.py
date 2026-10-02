"""Verify all distributed bytes against the supplied SHA-256 manifest."""
from pathlib import Path
import argparse,csv,hashlib
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();n=0
with (a.root/'MANIFEST_SHA256.csv').open() as f:
 for row in csv.DictReader(f):
  path=a.root/row['path'];assert path.is_file(),row['path'];assert path.stat().st_size==int(row['bytes']),row['path']
  assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path'];n+=1
print(f'{n} files passed byte-level verification.')
