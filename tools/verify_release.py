"""Offline release checks. No data files or API keys are needed; networking is blocked."""
import ast
import json
import os
from pathlib import Path
import re
import runpy
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def blocked(*args, **kwargs):
    raise AssertionError('Network access is forbidden during offline verification')


def main():
    urllib.request.urlopen = blocked
    scripts = list(ROOT.rglob('*.py'))
    for path in scripts:
        text = path.read_text(encoding='utf-8-sig')
        tree = ast.parse(text, filename=str(path))
        assert not re.search(r'(?:apikey_|sk-)[A-Za-z0-9_-]{16,}', text), path
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not re.match(r'^[A-Za-z]:[\\/]', node.value), path
    for path in ROOT.rglob('*.json'):
        obj = json.loads(path.read_text(encoding='utf-8-sig'))
        if path.name in ('api.json', 'api_key.json'):
            assert obj == {'api_key': 'YOUR API KEY'}, path
    manifest = json.loads((ROOT / 'release_manifest.json').read_text(encoding='utf-8'))
    assert all((ROOT / p).is_file() for p in manifest['copied_files'])
    original_scripts = [p for p in manifest['copied_files'] if p.endswith('.py')]
    for rel in original_scripts:
        runpy.run_path(str(ROOT / rel), run_name='release_import_check')
    # Exercise the actual converter with a complete small synthetic ARFF fixture.
    converter = runpy.run_path(str(ROOT / 'Univariate2018_arff/arff2csv.py'))
    with tempfile.TemporaryDirectory() as folder:
        src, dst = Path(folder) / 'tiny.arff', Path(folder) / 'tiny.csv'
        src.write_text('@relation tiny\n@attribute value numeric\n@attribute class {a,b}\n@data\n1,a\n2,b\n', encoding='utf-8')
        converter['convert_arff_to_csv'](src, dst)
        import csv
        with dst.open(encoding='utf-8-sig', newline='') as stream:
            assert list(csv.reader(stream)) == [['value', 'class'], ['1', 'a'], ['2', 'b']]
    # Placeholder keys must fail locally before any HTTP request.
    os.environ.pop('JEV_API_KEY', None)
    readers = [
        ('easyTPP/test_amazon.py', '_api_key', ()),
        ('easyTPP/easytpp_jev_common.py', '_api_key', ()),
        ('Univariate2018_arff/2/BeetleFly/test.py', 'load_api_key', ()),
        ('Univariate2018_arff/n/ACSF1_10/test.py', '_api_key', (ROOT / 'Univariate2018_arff/n/ACSF1_10',)),
        ('Multivariate2018_arff/Multivariate_arff/jev_multivariate_runner.py', '_api_keys', (ROOT,)),
    ]
    for rel, name, args in readers:
        namespace = runpy.run_path(str(ROOT / rel))
        try:
            namespace[name](*args)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f'Placeholder key accepted by {rel}')
    print(f'PASS: {len(scripts)} Python files parsed; {len(original_scripts)} original modules imported; credentials, paths, file inventory, and ARFF conversion checked offline.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
