# Benchmarking System One Models for Time-Series Decision Making: A Case Study of Jev

This release contains 173 original Python files for JEV experiments, configuration JSON files, EasyTPP sampling manifests, dataset descriptions, and the dataset directory structure. Raw data, historical predictions, archives, caches, and editor settings are excluded. Empty directories are preserved with `.gitkeep`. The original source directory has not been modified.

The project uses the official hosted JEV API for zero-shot classification. It includes neither model training nor model weights. The `easyTPP` directory contains event-type prediction experiments; it does not contain the full upstream EasyTPP training framework. All runtime dependencies are part of the Python standard library.

## 1. Directory Structure and Experiment Scope

| Directory | Default experiments | Input |
| --- | ---: | --- |
| `easyTPP/` | 5 | Test JSON files for Amazon, Retweet, StackOverflow, Taobao, and Taxi |
| `Multivariate2018_arff/Multivariate_arff/` | 30 | `<dataset>_TEST.arff` for each dataset |
| `Univariate2018_arff/2/` and `Univariate2018_arff/n/` | 128 total | `<dataset>_TEST.arff` for each dataset |
| `GunPoint/` | 1 | `GunPoint_TEST.txt`, with the label in the first column |

One additional JEV backup entry point, `GunPoint/test copy.py`, is retained but excluded from default batch runs. All entry points and their required inputs are listed in `benchmark_inventory.json`. The 164 unique required data files and their original SHA-256 checksums are listed in `data_manifest.json`. `release_manifest.json` records retained and excluded source files, including subsequent removals.

Run all commands below from this directory: the `code` directory containing this README. Experiment scripts locate inputs relative to their own locations, independently of the author's drive letters or working directory.

## 2. Install the Complete Conda Environment

Recommended installation:

```sh
conda env create -f environment.yml
conda activate jev-bench
```

Alternatively, use the Conda package specification directly:

```sh
conda create -n jev-bench --file conda-spec.txt -y
conda activate jev-bench
```

To install the complete runtime into an existing environment:

```sh
conda install -n jev-bench --file conda-spec.txt -y
```

Python is pinned to the validated version, `3.10.16`. No pip installation is required. Conda resolves Python's underlying dependencies automatically; those packages are not locked across platforms. Dependency resolution was checked with a Conda dry run on Windows. Other platforms have not been tested.

## 3. Prepare the Data

### EasyTPP: Download Pinned Versions Automatically

The official source is the [Hugging Face easytpp organization](https://huggingface.co/easytpp), published by the [EasyTPP project](https://github.com/ant-research/EasyTemporalPointProcess). All five local JSON files were compared with the corresponding official revisions and matched byte for byte. The downloader uses only the standard library and verifies SHA-256 checksums:

```sh
python tools/download_easytpp.py
python tools/prepare_data.py --verify --suite easyTPP
```

Pinned commits, download URLs, and checksums are recorded in `easytpp_sources.json`. The local StackOverflow filename retains the original project's spelling, `stockflow_test.json`. Do not rename it to `stackoverflow_test.json`.

| Official repository | Local path | Sequences | Event types |
| --- | --- | ---: | ---: |
| `easytpp/amazon` | `easyTPP/amazon_test.json` | 1851 | 16 |
| `easytpp/retweet` | `easyTPP/retweet_test.json` | 2000 | 3 |
| `easytpp/stackoverflow` | `easyTPP/stockflow_test.json` | 401 | 22 |
| `easytpp/taobao` | `easyTPP/taobao_test.json` | 500 | 17 |
| `easytpp/taxi` | `easyTPP/taxi_test.json` | 400 | 10 |

Each JSON file is a top-level array. Each record contains `dim_process`, `seq_idx`, `seq_len`, `type_event`, `time_since_start`, and `time_since_last_event`. Preserve the original record order: seeded sampling depends on it.

### UCR/UEA and GunPoint

Download the 2018 ARFF archives from the [official time-series classification data website](https://www.timeseriesclassification.com/):

- [Univariate2018_arff.zip](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_arff.zip)
- [Multivariate2018_arff.zip](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_arff.zip)

Extract the archives into a staging directory outside this repository. Replace `../downloaded-data` below with your extraction directory. The preparation tool first searches by relative path, then by filename and SHA-256. It can therefore map files from the official archive layout into this project's `2/`, `n/`, and class-count-suffixed directories:

```sh
python tools/prepare_data.py --source ../downloaded-data --suite Univariate2018_arff
python tools/prepare_data.py --source ../downloaded-data --suite Multivariate2018_arff
python tools/prepare_data.py --source ../downloaded-data --suite GunPoint
```

If you already have the original data tree, restore all required test inputs at once. Replace `../original-data` with that directory:

```sh
python tools/prepare_data.py --source ../original-data
```

The tool copies only required test inputs. It does not copy training sets, credentials, or historical outputs, and it refuses to overwrite existing files with different contents. You can also place files manually; exact target paths are listed in `data_manifest.json`. Examples:

```text
GunPoint/GunPoint_TEST.txt
Univariate2018_arff/2/BeetleFly/BeetleFly_TEST.arff
Univariate2018_arff/n/ArrowHead_3/ArrowHead_TEST.arff
Multivariate2018_arff/Multivariate_arff/BasicMotions/BasicMotions_TEST.arff
```

Check all inputs after preparation:

```sh
python tools/prepare_data.py --verify
python tools/run_benchmarks.py --check-data
```

If a checksum differs, check the dataset version, line endings, and file contents before proceeding. Only TEST files are required; model calls do not use TRAIN files. Dataset contents and descriptions remain subject to their providers' terms. The project's MIT license applies to project code.

## 4. Configure the Official JEV Service

Obtain an API key through the [official TypeSafe console](https://console.typesafe.ai/) and confirm that your account can access the original experiment's pinned model, `jev-1.13.0`. Requests follow the [official API documentation](https://api.typesafe.ai/redoc): `POST https://api.typesafe.ai/v1/systemone`, authenticated with `Authorization: Bearer <API_KEY>`. HTTP calls are already implemented; no additional SDK or local model deployment is required.

Environment variables are the recommended way to provide credentials. In PowerShell:

```powershell
$env:JEV_API_KEY = "YOUR API KEY"
```

In a Linux/macOS shell:

```sh
export JEV_API_KEY="YOUR API KEY"
```

Replace the placeholder with your real key before running experiments. Placeholder keys are rejected locally. Environment variables take precedence over JSON settings. Alternatively, fill in the relevant configuration file:

- `easyTPP/api.json`
- `Multivariate2018_arff/Multivariate_arff/api_key.json`
- `Univariate2018_arff/api_key.json`

Each template contains `{"api_key": "YOUR API KEY"}`. The GunPoint scripts and the EasyTPP common module also support `JEV_API_KEY`. The multivariate runner additionally supports `JEV_FALLBACK_API_KEY` within its existing token-limit fallback logic. Keep placeholder configurations in the published repository and use environment variables for experiments.

`API_ENDPOINT`, `MODEL`, and script-specific batch sizes retain the original experiment settings. If the official service no longer supports the pinned model for your account, check available models before changing `MODEL`. A run using a different model version is a separate experiment and does not strictly reproduce the original results.

## 5. Reproduce the Experiments

First run the offline release checks. They require neither data nor credentials and block network requests:

```sh
python tools/verify_release.py
```

After preparing data and credentials, list the default experiments, check their inputs, and run them:

```sh
python tools/run_benchmarks.py --list
python tools/run_benchmarks.py --check-data
python tools/run_benchmarks.py
```

The batch runner executes 164 default entry points sequentially, preserving each script's defaults. Exit codes are saved in `benchmark_runs.json`. It continues after individual failures and returns a nonzero exit code if any experiment fails. Actual experiments call the remote official service and consume account quota. Runtime depends on sequence counts, sequence lengths, service limits, and retries.

You can also run individual suites or select a dataset by matching its entry-point path:

```sh
python tools/run_benchmarks.py --suite easyTPP
python tools/run_benchmarks.py --suite Multivariate2018_arff
python tools/run_benchmarks.py --suite Univariate2018_arff
python tools/run_benchmarks.py --suite GunPoint
python tools/run_benchmarks.py --suite Univariate2018_arff --match BeetleFly
```

### EasyTPP Parameters and Preflight Checks

By default, each test set is sampled with `seed=42` to select 100 sequences. Each batch contains targets from at most 10 distinct sequences. Starting at the second event of each sequence, the model predicts the target event's type using its true event history and known target time. This evaluates event-type prediction conditioned on the target time; it does not evaluate prediction of an unknown next-event time.

```sh
python easyTPP/test_amazon.py --num-sequences 100 --batch-size 10 --seed 42 --dry-run
python easyTPP/test_amazon.py --num-sequences 100 --batch-size 10 --seed 42
python easyTPP/test_retweet.py --num-sequences 100 --batch-size 10 --seed 42
python easyTPP/test_stockflow.py --num-sequences 100 --batch-size 10 --seed 42
python easyTPP/test_taobao.py --num-sequences 100 --batch-size 10 --seed 42
python easyTPP/test_taxi.py --num-sequences 100 --batch-size 10 --seed 42
```

`--dry-run` checks sampling, batching, and history construction without calling the API. All five EasyTPP test scripts support `--output-dir`. Use a new output directory for each reproduction run to avoid overwriting existing outputs or reference sampling manifests.

The retained `*_selected_sequences.json` files contain sampling parameters, sequence indices, and lengths, without raw event observations. They serve as references for the original selections. Each run regenerates its selection using the seed rather than reading a reference manifest directly.

Reference target counts are Amazon 4580, Retweet 10383, StackOverflow 6864, Taobao 5636, and Taxi 3598. Identical parameters, input contents, and record order should reproduce the same sequence selection.

### Individual Univariate, Multivariate, and GunPoint Entry Points

```sh
python Univariate2018_arff/2/BeetleFly/test.py
python Univariate2018_arff/n/ArrowHead_3/test.py
python Multivariate2018_arff/Multivariate_arff/BasicMotions/test.py
python GunPoint/test.py
```

These scripts read the entire TEST set by default and use the `BATCH_SIZE`, prompts, and class definitions in their source files. They do not share a `--dry-run` option. Use the batch runner's `--check-data` option first to check input availability.

## 6. Outputs and Evaluation

Predictions and metrics are written into the existing `jev_*_output` directory for each dataset. EasyTPP uses `jev_*_sampled_output`. Ground-truth labels are used to compute metrics locally; target labels are excluded from the model's observation inputs. Original prompts, class semantics, retries, and token-limit batch splitting are preserved.

- Univariate/multivariate: `*_TEST_jev_predictions.csv` and `*_TEST_acc_auc.txt`.
- EasyTPP: `*_next_event_predictions.csv`, `*_next_event_metrics.txt`, and sampling JSON manifests. Metrics include ACC, event-type error rate, macro one-vs-rest AUC, Macro F1, and mean per-sequence ACC.
- The main GunPoint script retains its original output filenames and terminal metric format.

Summarize results:

```sh
python easyTPP/summarize_acc_auc.py
python Multivariate2018_arff/summarize_acc_auc.py
python Univariate2018_arff/2/summarize_acc_auc.py
```

The generic univariate summarizer can also process `n/`:

```sh
python Univariate2018_arff/2/summarize_acc_auc.py --root Univariate2018_arff/n
```

AUC uses the original scripts' rank-based implementations. Binary positive classes and multiclass label definitions are specified in each script's configuration. Missing-class behavior is also preserved: some multiclass scripts return `N/A`, while the multivariate runner averages over classes with defined AUC values. Review the relevant implementation and class coverage when interpreting results.

## 7. Conversion Utility and JEV Backup Entry Point

The univariate ARFF-to-CSV utility includes a fix for CSV header generation and accepts explicit input and output directories:

```sh
python Univariate2018_arff/arff2csv.py --input-dir Univariate2018_arff/n/ArrowHead_3 --output-dir ../ArrowHead_csv
```

Default classification scripts read ARFF directly; CSV conversion is optional.

`GunPoint/test copy.py` is a retained JEV backup experiment script. It is excluded from default batch runs. Run it manually, quoting the filename containing a space, or pass `--include-alternatives` to the batch runner to include it.

## 8. Validation Scope and Reproducibility Limits

The 173 retained original modules were validated by offline import. All 165 retained experiment entry points passed input-reading checks: ARFF/TXT readers used the original headers and first two samples, while EasyTPP readers processed complete test JSON structures. All five official EasyTPP test files matched the original local files byte for byte. See [VALIDATION.md](VALIDATION.md) for details.

No real API keys were used to execute online inference, and full-dataset remote model metrics were not recomputed. Code and data preparation have been checked; service availability, account permissions, model access, and actual responses still require verification with a live account. A pinned remote model name may still be affected by service-side changes, so numerical results cannot be guaranteed to match historical runs exactly. Record the run date, data checksums, actual returned model version, batch sizes, seed, and generated outputs.

## 9. License and Sources

Project code is released under the [MIT License](LICENSE), with copyright attributed to WANG. Data, original dataset descriptions, and external services retain their respective licenses and terms.

- [Official TypeSafe API documentation](https://api.typesafe.ai/redoc)
- [Official EasyTPP repository](https://github.com/ant-research/EasyTemporalPointProcess) and [dataset format documentation](https://ant-research.github.io/EasyTemporalPointProcess/user_guide/dataset.html)
- [Official UCR/UEA dataset website](https://www.timeseriesclassification.com/)
