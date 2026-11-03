"""Download the five pinned official EasyTPP test JSONs and verify SHA-256."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = json.loads((ROOT / 'easytpp_sources.json').read_text(encoding='utf-8'))
    for row in rows:
        target = ROOT / 'easyTPP' / row['local_file']
        if target.exists():
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != row['download_sha256']:
                raise RuntimeError(f'Refusing to overwrite different file: {target.name}')
            print(f'Already verified: {target.name}')
            continue
        print(f'Downloading {row["dataset"]} @ {row["revision"]}', flush=True)
        with urllib.request.urlopen(row['url'], timeout=180) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != row['download_sha256']:
            raise RuntimeError(f'Checksum mismatch: {target.name}')
        target.write_bytes(content)
        print(f'Verified: {target.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
