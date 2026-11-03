# JEV Bench：时序分类与事件类型预测

本目录是 `jev_bench/时序分类` 的开源整理版本，保留 173 个与 JEV 实验相关的原始 Python 文件、配置 JSON、EasyTPP 抽样索引、数据集说明和数据目录结构。原始数据、历史预测结果、压缩包、缓存和编辑器配置不随代码发布。空目录用 `.gitkeep` 保留，原目录未修改。

项目调用官方托管的 JEV API 做零样本分类，不包含模型训练或模型权重。本项目的 `easyTPP` 是事件类型预测实验脚本，并非完整的上游 EasyTPP 训练框架。全部运行依赖均为 Python 标准库。

## 1. 目录与实验范围

| 目录 | 默认实验数 | 输入 |
| --- | ---: | --- |
| `easyTPP/` | 5 | Amazon、Retweet、StackOverflow、Taobao、Taxi 测试 JSON |
| `Multivariate2018_arff/Multivariate_arff/` | 30 | 每个数据集的 `<名称>_TEST.arff` |
| `Univariate2018_arff/2/`、`Univariate2018_arff/n/` | 共 128 | 每个数据集的 `<名称>_TEST.arff` |
| `GunPoint/` | 1 | `GunPoint_TEST.txt`，标签在第一列 |

另外保留 1 个 JEV 备份入口：GunPoint 的 `test copy.py`，默认不参加批量运行。所有入口与所需输入见 `benchmark_inventory.json`；所需的 164 个唯一数据文件及原始 SHA-256 见 `data_manifest.json`。`release_manifest.json` 记录原目录中复制及排除的文件。

下述命令均从本目录（包含此 README 的 `code`）运行。代码依据脚本位置定位数据，不依赖作者电脑的盘符或工作目录。

## 2. 安装完整 Conda 环境

推荐安装方式：

```sh
conda env create -f environment.yml
conda activate jev-bench
```

也可使用可直接交给 Conda 的依赖列表：

```sh
conda create -n jev-bench --file conda-spec.txt -y
conda activate jev-bench
```

如已存在环境，可直接安装完整运行依赖：

```sh
conda install -n jev-bench --file conda-spec.txt -y
```

Python 固定为已验证的 `3.10.16`，无需执行 pip 安装。Conda 会自动解析 Python 的底层依赖；这些底层包未跨平台锁定。整理时在 Windows 上通过 Conda dry-run 解析；其他平台尚未实测。

## 3. 准备数据

### EasyTPP：固定版本自动下载

官方来源是 [EasyTPP 项目](https://github.com/ant-research/EasyTemporalPointProcess) 发布的 [Hugging Face easytpp 数据集](https://huggingface.co/easytpp)。本项目五个 JSON 已与官方对应版本逐字节核对，一致。下载脚本只使用标准库，并验证 SHA-256：

```sh
python tools/download_easytpp.py
python tools/prepare_data.py --verify --suite easyTPP
```

每个数据仓库的固定 commit、下载地址和校验值均记录在 `easytpp_sources.json`。StackOverflow 的本地文件名保留原项目拼写 `stockflow_test.json`，请勿改成 `stackoverflow_test.json`。

| 官方仓库 | 本地位置 | 序列数 | 类型数 |
| --- | --- | ---: | ---: |
| `easytpp/amazon` | `easyTPP/amazon_test.json` | 1851 | 16 |
| `easytpp/retweet` | `easyTPP/retweet_test.json` | 2000 | 3 |
| `easytpp/stackoverflow` | `easyTPP/stockflow_test.json` | 401 | 22 |
| `easytpp/taobao` | `easyTPP/taobao_test.json` | 500 | 17 |
| `easytpp/taxi` | `easyTPP/taxi_test.json` | 400 | 10 |

JSON 顶层为数组，每条记录包含 `dim_process`、`seq_idx`、`seq_len`、`type_event`、`time_since_start` 和 `time_since_last_event`。保持原始记录顺序，随机抽样依赖该顺序。

### UCR/UEA 与 GunPoint

从 [时间序列分类官方数据站](https://www.timeseriesclassification.com/) 下载 2018 版 ARFF 数据：

- [Univariate2018_arff.zip](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_arff.zip)
- [Multivariate2018_arff.zip](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_arff.zip)

解压至仓库外的数据暂存目录。将下面命令中的 `../downloaded-data` 替换为自己的解压目录；工具先按相对路径查找，再按文件名和 SHA-256 查找，因此可接受官方压缩包解压后的布局，并自动放入本项目的 `2/`、`n/` 和带类别数后缀的目录：

```sh
python tools/prepare_data.py --source ../downloaded-data --suite Univariate2018_arff
python tools/prepare_data.py --source ../downloaded-data --suite Multivariate2018_arff
python tools/prepare_data.py --source ../downloaded-data --suite GunPoint
```

如果已经拥有原始 `时序分类` 数据目录，也可一次恢复所有必要测试文件：

```sh
python tools/prepare_data.py --source ../时序分类
```

恢复工具只复制本次运行所需的测试输入，不复制训练集、密钥或历史结果；不会覆盖内容不同的已有文件。可手动放置数据，准确目标路径见 `data_manifest.json`。例：

```text
GunPoint/GunPoint_TEST.txt
Univariate2018_arff/2/BeetleFly/BeetleFly_TEST.arff
Univariate2018_arff/n/ArrowHead_3/ArrowHead_TEST.arff
Multivariate2018_arff/Multivariate_arff/BasicMotions/BasicMotions_TEST.arff
```

完成后检查所有数据：

```sh
python tools/prepare_data.py --verify
python tools/run_benchmarks.py --check-data
```

校验不一致时应核对数据版本、换行和文件内容，不要直接删除校验约束。这里只需要 TEST 文件，模型调用不使用 TRAIN。原始数据及其描述的使用条款由各自提供方规定，本项目 MIT 许可适用于项目代码。

## 4. 配置 JEV 官方服务

通过 [TypeSafe 官方控制台](https://console.typesafe.ai/) 获取可用 API key，并确认账户可以访问原实验指定的模型 `jev-1.13.0`。实际请求遵循 [官方 API 文档](https://api.typesafe.ai/redoc)：`POST https://api.typesafe.ai/v1/systemone`，使用 `Authorization: Bearer <API_KEY>`。代码已包含 HTTP 调用逻辑，无需额外 SDK 或本地模型部署。

推荐用环境变量传入凭据。PowerShell：

```powershell
$env:JEV_API_KEY = "YOUR API KEY"
```

Linux/macOS shell：

```sh
export JEV_API_KEY="YOUR API KEY"
```

运行前将占位符替换为自己的真实 key。占位符会在本地被拒绝。环境变量优先于 JSON；也可填写相应配置：

- `easyTPP/api.json`
- `Multivariate2018_arff/Multivariate_arff/api_key.json`
- `Univariate2018_arff/api_key.json`

所有模板内容为 `{"api_key": "YOUR API KEY"}`。GunPoint 原始脚本和 EasyTPP 通用模块也支持 `JEV_API_KEY`。多变量 runner 额外支持 `JEV_FALLBACK_API_KEY`，仅在原有 token 限制回退逻辑中使用。发布仓库应保持占位符配置，用环境变量运行实验。

`API_ENDPOINT`、`MODEL` 和各脚本原有批大小保留原实验设置。若官方账户不再支持固定模型，须先核对服务端可用模型；修改 `MODEL` 后的运行属于不同模型版本的实验，不能视为原结果的严格复现。

## 5. 完整复现流程

先进行无需数据、无需密钥、禁止网络请求的发布检查：

```sh
python tools/verify_release.py
```

准备完数据和凭据后，可列出全部默认实验入口，再统一执行：

```sh
python tools/run_benchmarks.py --list
python tools/run_benchmarks.py --check-data
python tools/run_benchmarks.py
```

批量工具按顺序执行 164 个默认入口，保留每个脚本原有默认参数，退出码记录到 `benchmark_runs.json`。单项失败后继续其他项目，最终有失败时返回非零退出码。实际运行会调用官方远程服务并消耗账户额度；耗时取决于序列数、长度、服务限额及重试情况。

也可按套件或数据集运行：

```sh
python tools/run_benchmarks.py --suite easyTPP
python tools/run_benchmarks.py --suite Multivariate2018_arff
python tools/run_benchmarks.py --suite Univariate2018_arff
python tools/run_benchmarks.py --suite GunPoint
python tools/run_benchmarks.py --suite Univariate2018_arff --match BeetleFly
```

### EasyTPP 参数与预检查

每个测试集默认使用 `seed=42` 抽取 100 条序列，每批最多来自 10 条不同序列。每条序列从第二个事件开始，使用真实历史事件及已知目标时间预测目标事件类型。因此本实验评估的是给定目标时间的事件类型预测，并未评估未知下一事件时间的预测。

```sh
python easyTPP/test_amazon.py --num-sequences 100 --batch-size 10 --seed 42 --dry-run
python easyTPP/test_amazon.py --num-sequences 100 --batch-size 10 --seed 42
python easyTPP/test_retweet.py --num-sequences 100 --batch-size 10 --seed 42
python easyTPP/test_stockflow.py --num-sequences 100 --batch-size 10 --seed 42
python easyTPP/test_taobao.py --num-sequences 100 --batch-size 10 --seed 42
python easyTPP/test_taxi.py --num-sequences 100 --batch-size 10 --seed 42
```

`--dry-run` 验证抽样、分批和历史构造，不调用 API。所有脚本支持 `--output-dir`，建议复现实验指定一个新目录，避免覆盖已有输出及抽样索引。已保留的 `*_selected_sequences.json` 只有抽样种子、索引和长度，不含事件原始观测；它们是原实验抽样参考。默认每次根据固定种子重新生成抽样，不直接读取参考索引。

原实验参考的目标数分别为 Amazon 4580、Retweet 10383、StackOverflow 6864、Taobao 5636、Taxi 3598。参数、数据和顺序相同时应得到相同抽样索引。

### 单变量、多变量与 GunPoint 单项入口

```sh
python Univariate2018_arff/2/BeetleFly/test.py
python Univariate2018_arff/n/ArrowHead_3/test.py
python Multivariate2018_arff/Multivariate_arff/BasicMotions/test.py
python GunPoint/test.py
```

这些脚本默认读取整个 TEST 集，使用各自源码中的 `BATCH_SIZE`、提示词和类别定义。它们不支持统一的 `--dry-run` 参数；可先用批量工具的 `--check-data` 检查输入是否齐全。

## 6. 输出与评估

预测与指标写入各数据集下原有 `jev_*_output` 目录；EasyTPP 写入 `jev_*_sampled_output`。输入标签用于本地计算指标，脚本的观测输入不包含目标标签。原始提示词、类别语义、重试与 token 限制拆分逻辑均保留。

- 单变量/多变量：`*_TEST_jev_predictions.csv`、`*_TEST_acc_auc.txt`。
- EasyTPP：`*_next_event_predictions.csv`、`*_next_event_metrics.txt`、抽样索引 JSON。指标包含 ACC、事件类型错误率、macro one-vs-rest AUC、Macro F1、逐序列平均 ACC 等。
- GunPoint 主脚本保留其原有输出命名与终端指标格式。

结果汇总：

```sh
python easyTPP/summarize_acc_auc.py
python Multivariate2018_arff/summarize_acc_auc.py
python Univariate2018_arff/2/summarize_acc_auc.py
```

通用单变量汇总器也可用于 `n/`：

```sh
python Univariate2018_arff/2/summarize_acc_auc.py --root Univariate2018_arff/n
```

AUC 使用原脚本的秩统计实现，二分类的正类与多分类的类别定义由各脚本配置决定。缺少某类别时的处理保留原实现：部分多分类脚本返回 `N/A`，多变量 runner 对有定义的类别 AUC 取平均，因此应同时查看具体脚本与类别覆盖情况。

## 7. 辅助转换与 JEV 备份入口

单变量 ARFF 转 CSV 工具已修复表头写入问题，可指定目录：

```sh
python Univariate2018_arff/arff2csv.py --input-dir Univariate2018_arff/n/ArrowHead_3 --output-dir ../ArrowHead_csv
```

默认分类脚本读取 ARFF，不需要先转 CSV。

`GunPoint/test copy.py` 是原项目保留的 JEV 备份实验脚本，默认不参加批量运行。可手动运行该脚本，或使用批量工具的 `--include-alternatives` 将其包含在运行清单中。

## 8. 验证范围与复现边界

整理时已验证保留的 173 个原始模块可离线导入，165 个实验入口能读取原始测试输入（ARFF/TXT 抽取前两行，EasyTPP 读取完整结构），并核对五份 EasyTPP 官方测试数据与原文件逐字节一致。详见 `VALIDATION.md`。

未使用原作者密钥执行在线推理，未重新计算完整数据集的远程模型指标。代码和数据准备已验证，但官方服务可用性、账户权限、模型版本以及在线返回结果需要在实际账户上运行确认。远程模型即使固定名称，也可能受服务端更新影响，不能保证数值与历史运行完全一致；请保存运行日期、数据校验值、模型实际返回版本、批大小、种子及新生成输出。

## 9. 许可证与来源

项目代码使用 [MIT License](LICENSE)，版权署名 WANG。数据、原数据集描述以及外部服务保留各自许可与使用条款。

- [TypeSafe 官方 API 文档](https://api.typesafe.ai/redoc)
- [EasyTPP 官方代码](https://github.com/ant-research/EasyTemporalPointProcess) 与 [数据格式文档](https://ant-research.github.io/EasyTemporalPointProcess/user_guide/dataset.html)
- [UCR/UEA 官方数据站](https://www.timeseriesclassification.com/)
