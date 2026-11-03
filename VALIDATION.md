# Release Validation Record

Validation date: 2026-11-03 (Asia/Shanghai). Platform: Windows, Python 3.10.16, Conda 25.5.0.

## Passed Checks

| Check | Result |
| --- | --- |
| Retained original Python files | 173 JEV-related files; 3 experiments using another model were removed as requested |
| Python syntax and module imports | All 177 release Python files parsed successfully; all 173 retained original modules imported offline |
| Data-reading entry points | 165 passed: ARFF/TXT readers used original headers and the first two samples; EasyTPP readers used complete test JSON files |
| Official EasyTPP data consistency | All 5 pinned revisions of `test.json` matched the original local files byte for byte |
| EasyTPP dry runs | All 5 passed; sampled sequence indices matched the original reference manifests with seed 42, 100 sequences, and batch size 10 |
| EasyTPP target counts | Amazon 4580; Retweet 10383; StackOverflow 6864; Taobao 5636; Taxi 3598 |
| Conda installation plan | `conda create --dry-run --name jev-bench-release-check --file conda-spec.txt --json` returned `success=true` |
| Credential removal | API-key JSON files and source credentials use `YOUR API KEY`; no long `apikey_` or `sk-` credential patterns were found |
| Machine-specific absolute paths | No Windows drive-letter absolute paths were found in Python string constants; inputs are located relative to script locations |
| Placeholder protection | Credential readers for EasyTPP, its common module, BeetleFly, univariate experiments, and multivariate experiments rejected placeholder keys |
| ARFF-to-CSV conversion | Column names and both data rows were correct for a synthetic two-row ARFF fixture |
| Batch preflight | Discovered 164 default experiments; the data-free release correctly reported 164 missing inputs and returned exit code 2 |
| Data restoration tool | File discovery, restoration, SHA-256 verification, and refusal to overwrite different contents passed in temporary directories |
| EasyTPP downloader | All 5 filename mappings and checksums passed using verified official files as simulated download responses; the actual official files were read separately during consistency verification |
| Data-free release contents | No ARFF, TS, TSV, CSV, ZIP, database, bytecode, raw test JSON, or TRAIN/TEST TXT files were packaged |
| Markdown documentation language | All 45 Markdown files were checked; the project README and validation record were translated into English, and the existing dataset descriptions were already in English |

Temporary data checks ran in separate temporary directories. The release retains empty data directories and sampling manifests, without the observations used during validation.

## Changes Made

- Removed 3 non-JEV model experiment scripts and their empty output directories; updated the entry-point inventory, summary examples, and README.
- Preserved the retained scripts' directory layout, prompts, class definitions, model version, and original batch sizes.
- Replaced hardcoded credentials with placeholders and added environment-variable support.
- Replaced the ARFF conversion utility's machine-specific data paths with paths based on its script location and added directory arguments.
- Fixed the ARFF converter's attempt to write the CSV header before lazily reading the ARFF header.
- Deferred BeetleFly credential loading until `main`, allowing module imports without real credentials.
- Added environment specifications, the MIT license, reproduction instructions, data checksums, batch entry points, and data preparation tools.
- Provided English Markdown documentation throughout the release.

## Not Yet Validated

- No online inference was performed with original or newly supplied real API keys. Full-dataset remote model metrics were not rerun.
- ARFF/TXT input-reading checks covered the first two samples for each entry point, rather than every data row.
- Conda dependency resolution was checked without creating a new environment. Offline code checks used the same Python version already installed on the machine.
- Linux and macOS execution has not been tested.
- The official UCR/UEA archives were not downloaded in full again for comparison. Their manifest checksums identify the author's existing original files. All five official EasyTPP JSON files were compared in full.

After preparing data and account access, follow the README to run a complete online reproduction and save separate outputs. The current checks do not guarantee service availability or identical remote model metrics across runs.
