"""Restore required test files from a separately obtained data tree, or verify them."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Root with the same relative layout as this repository')
    parser.add_argument('--verify', action='store_true', help='Verify local data against original SHA-256')
    parser.add_argument('--suite', default='all', choices=['all', 'easyTPP', 'Multivariate2018_arff', 'Univariate2018_arff', 'GunPoint'])
    args = parser.parse_args()
    if not args.verify and args.source is None:
        parser.error('Provide --source or --verify')
    records = json.loads((ROOT / 'data_manifest.json').read_text(encoding='utf-8'))
    failures = []
    candidates_by_name = None
    for row in records:
        if args.suite != 'all' and Path(row['path']).parts[0] != args.suite:
            continue
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe manifest path')
        target = ROOT / rel
        if args.verify:
            source = target
        else:
            source = args.source.resolve() / rel
            if not source.is_file():
                if candidates_by_name is None:
                    candidates_by_name = {}
                    for candidate in args.source.resolve().rglob('*'):
                        if candidate.is_file():
                            candidates_by_name.setdefault(candidate.name, []).append(candidate)
                candidates = candidates_by_name.get(rel.name, [])
                source = next((p for p in candidates if p.stat().st_size == row['bytes'] and sha256(p) == row['sha256']), source)
        if not source.is_file():
            failures.append(f'Missing: {rel.as_posix()}')
            continue
        if source.stat().st_size != row['bytes'] or sha256(source) != row['sha256']:
            failures.append(f'Checksum mismatch: {rel.as_posix()}')
            continue
        if not args.verify and source.resolve() != target.resolve():
            if target.exists():
                if sha256(target) != row['sha256']:
                    failures.append(f'Refusing to overwrite different file: {rel.as_posix()}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for failure in failures:
        print(failure, file=sys.stderr)
    print('Data preparation/verification passed.' if not failures else f'{len(failures)} data problem(s).')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
