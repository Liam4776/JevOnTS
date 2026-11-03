"""List, preflight, or run the preserved benchmark entry points."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['all', 'easyTPP', 'Multivariate2018_arff', 'Univariate2018_arff', 'GunPoint'], default='all')
    parser.add_argument('--match', default='', help='Case-insensitive substring of entry path')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--check-data', action='store_true', help='Check required files without calling APIs')
    parser.add_argument('--include-alternatives', action='store_true', help='Include the backup GunPoint JEV script')
    args = parser.parse_args()
    rows = json.loads((ROOT / 'benchmark_inventory.json').read_text(encoding='utf-8'))
    rows = [r for r in rows if (args.suite == 'all' or r['suite'] == args.suite)
            and args.match.lower() in r['entry'].lower()
            and (args.include_alternatives or not r['alternative'])]
    if not rows:
        parser.error('No matching benchmarks')
    if args.list:
        for r in rows:
            print(r['entry'])
        print(f'{len(rows)} benchmark(s)')
        return 0
    missing = [r['data'] for r in rows if not (ROOT / r['data']).is_file()]
    if missing:
        for path in missing:
            print(f'Missing data: {path}', file=sys.stderr)
        return 2
    if args.check_data:
        print(f'Data present for {len(rows)} benchmark(s). No API calls made.')
        return 0
    results = []
    for r in rows:
        print(f"Running {r['entry']}", flush=True)
        code = subprocess.run([sys.executable, str(ROOT / r['entry'])], cwd=ROOT).returncode
        results.append({'entry': r['entry'], 'returncode': code})
    (ROOT / 'benchmark_runs.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    return int(any(r['returncode'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
