# Semantic Search Demo

This is a minimalistic demo on semantic search based on Rust ecosystem (pipeline), and it is basically the
[candle](https://github.com/huggingface/candle)'s
[official demo on BERT](https://github.com/huggingface/candle/blob/main/candle-examples/examples/bert/main.rs)
with few modifications.

## Requirements & Setup

Make sure following components are installed, or change `Makefile` accordingly.

- `make`
- `curl`
- `unzip`
- `docker`
- `build-essentials` or equivalent for `rustc`
- rust toolchain
- `CUDA` (if needed)

Then, execute

```shell
make init
```

This will download dataset from kaggle, pull&start database container at port `6333` and `6334` (
check [Qdrant document](https://qdrant.tech/documentation/quickstart/)
for more details).

Next, build the project (Optional)

```shell
# if you need Metal device
cargo b --features metal
 
# if you have CUDA device
cargo b --features cuda 

# or without GPU support
cargo b --features metal
```

## Usage

```
Usage: semantic_search_demo [OPTIONS]

Options:
  -c, --cpu              Use CPU
  -d, --db-uri <DB_URI>  Qdrant database uri [default: http://localhost:6334]
  -i, --init-db          Initialize Qdrant database
  -e, --embed            Embed dataset
  -q, --query            Query REPL
  -h, --help             Print help
  -V, --version          Print version
```

## Example

```shell
cargo r -r --features metal -- -ieq
```

This will: initialize a Qdrant Collection, embed the dataset (will normally take a while), and enter query REPL.

```shell
cargo r --features metal -- -ieq
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.25s
     Running `target/debug/semantic_search_demo -ieq`
config.json [00:00:00] [███████████████████████████] 645 B/645 B 4.90 KiB/s (0s)
tokenizer.json [00:00:00] [█████████████████] 8.66 MiB/8.66 MiB 18.19 MiB/s (0s)
model.safetensors [00:00:14] [██████████] 448.84 MiB/448.84 MiB 30.23 MiB/s (0s)
████████████████████████████████████████████████████████████████████ 94723/94723
Elapsed: 931.442728s
query> 测试
Num(202112230404) (0.58356416)
<"张家口冬奥村（冬残奥村）完成全要素测试 ">
"\u{3000}\u{3000}12月21日至22日，张家口冬奥村（冬残奥村）举行了全要素测试。\n\u{3000}\u{3000}本次测试共有2493人参加，共设置欢迎接待、餐饮保障、核酸采样、商业服务、开闭幕式演练等8个常规科目27个场景，设置运动员意外受伤急诊、火情扑救和人员疏散等4个应急演练科目。\n\u{3000}\u{3000}据悉，张家口冬奥村计划于2022年1月23日预开村，1月27日正式开村，3月16日闭村，共运行53天。\n\u{3000}\u{3000}图为冬奥村商业广场的特许商品店内，参测人员正在购买冬奥商品。\n\u{3000}\u{3000}本报记者\u{a0}\u{a0}张武军摄影报道\n"

Num(202111210203) (0.5608925)
<"北京冬奥村（冬残奥村）开展场馆运行测试 ">
"\u{3000}\u{3000}本报北京11月20日电\u{a0}\u{a0}（记者卢涛）北京冬奥村（冬残奥村）20日开展场馆运行测试，全面检验赛前运行状态。\n\u{3000}\u{3000}运行测试以能源设备供应为基础，检测居住区、运行区、广场区内各业务领域赛时运行空间的电力供应、技术信号覆盖、热水供应等设施设备运行情况，设置代表团注册会议、交通场站流线、抵离流程、欢迎接待等九大科目，模拟赛时重要场景。\n\u{3000}\u{3000}本次测试人员规模共计1630人，参测人员包括有奥运（残奥）经历的运动员和官员、残疾人士及无障碍专家、冬奥村建设者、志愿者等。\n"

Num(202302120303) (0.5560887)
<"集通铁路电气化改造 开展动态检测 ">
"\u{3000}\u{3000}2月10日，集通（内蒙古集宁—通辽）铁路电气化改造工程先开段动态检测试验圆满完成。动态检测是在工程静态验收合格后，采用综合检测列车和相关检测设备在规定测试速度下对全线各系统进行综合测试。\n\u{3000}\u{3000}图为10日，检测列车经过集通铁路兴和至化德区间。\n\u{3000}\u{3000}孙江昆摄（影像中国）\n"

Elapsed: 42.450917ms
query>
```

> [!NOTE]
> Do not initialize database twice, it will cause runtime-error.
> Potential solutions:
> - Remove the `-i` flag.
> - Goto database dashboard(http://localhost:6333/dashboard) and delete the collection.

> [!NOTE]
> All models downloaded by candle (hg_hub) will be stored at `~/.cache/huggingface/hub/`.

## Acknowledgement

- ML Framework used: [candle](https://github.com/huggingface/candle)
- BERT model
  used: [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- Dataset used: [People's Daily News](https://www.kaggle.com/datasets/concyclics/renmindaily)
- Vector Database: [Qdrant](https://github.com/qdrant/qdrant)