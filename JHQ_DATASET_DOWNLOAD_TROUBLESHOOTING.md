# JHQ 数据集下载踩坑记录

`scripts/download_jhq_datasets.py`——下载 JHQ 论文剩余4个数据集(openai3-1536、openai3-3072、bge-m3、stella-trec24,另外 Vogue-768、Arxiv-Abstracts-768 之前已下好)——过程中依次遇到的问题,按时间顺序记录,每条都是问题→原因→修法。环境:AutoDL 容器,国内网络,系统盘30G、数据盘另外挂载。

## 1. 内存爆掉(OOM)

**现象**:openai3-1536 下载进程静默消失,日志没有任何报错,`dmesg` 特征符合被系统 OOM killer 杀掉。

**原因**:脚本一开始就预分配整个 `(target_rows, dim)` 的 float32 数组再逐行填充——openai3-1536 约6GB,而 bge-m3(10M×1024维)要41GB、stella-trec24(17.8M×1024维)要73GB,远超机器内存。

**修法**:改成边流式下载边直接写入 `base.fvecs`/`query.fvecs`,不在内存里攒完整数组。

## 2. 数据集配置写错

**现象**:如果照搬 JHQ 论文写的规模去配置,下出来的数据跟论文对不上。

**原因**:
- BGE-M3 论文说的"10M"实际是 **Italian(`it`)配置**(10,092,524条),不是默认以为的 English(`en`,4700万条)。
- Stella-TREC24 根本没有 `train` 这个 split,真正的 split 是 `corpus`(1780万条,作为 base)和 `test_query`(65条,官方查询集,脚本目前仍是自己从 corpus 里抽样做 query,没直接用这个)。

**修法**:逐个查证官方数据集页面的 README/schema,改成正确的 `config`/`split`。

## 3. Ground truth 计算会撑爆内存/显存

**现象**:用 FAISS 建一个装下全部 base 向量的 flat index 来算精确近邻,理论上 bge-m3 要41GB、stella-trec24 要73GB——超过32G显卡,也大概率超过系统内存。

**修法**:换成纯 numpy 的分块暴力扫描(`compute_ground_truth_tiled`)——把 base 向量分批读回来,每批只跟 query 算一次距离矩阵、维护一个全局 top-k,内存峰值只跟"一批"的大小有关,跟数据总量无关,而且召回依然是精确的(暴力搜索),不是近似。后来因为这台机器内存偏紧,分块大小从20万预防性调小到2万。

## 4. `aiohttp` 不认代理

**现象**:用 `datasets.load_dataset(streaming=True)` 读 parquet,哪怕用 `curl` 测试确认代理(AutoDL 的 network_turbo)本身是通的,Python 这边还是一直握手超时。

**原因**:`datasets` 库流式读 parquet 走的是 `aiohttp` 发 HTTP range 请求,而 `aiohttp.ClientSession` **默认不读取 `http_proxy`/`https_proxy` 这两个环境变量**(跟 `requests`、`curl` 的行为不一样),代理设了也没用。

**修法**:换成 `huggingface_hub.hf_hub_download()`(基于 `requests`,正常认代理)。

## 5. `requests` 下载卡死不报错

**现象**:换成 `hf_hub_download()` 后能开始下载,但中途卡在固定字节数,几十秒内传输速率完全是0,不报任何异常,靠反复 `du -sb` 前后对比才确认是真卡死,不是看错。

**修法**:不再用任何 Python HTTP 库,直接 `subprocess` 调用系统的 `curl` 去下载文件本体,本地用 `pyarrow` 读取。

## 6. 直连 huggingface.co 不通

**现象/确认**:`curl` 干净测试(不带任何代理)直连 huggingface.co,4.2秒直接连接超时,确认是真不通,不是偶发。`hf-mirror.com` 和 AutoDL 自带的 `network_turbo` 代理分别单独测试都是通的(0.5-0.9秒)。

**修法**:改用 `hf-mirror.com` 直连下载文件,不依赖 AutoDL 的代理。

## 7. 缓存目录写错盘,系统盘被写满

**现象**:系统盘(`/`,固定30G)被写到100%,大量诡异行为(比如后面第8条的数据丢失)由此引发。

**原因**:下载用的原始 parquet 分片缓存目录写在了 `/root/.cache/jhq_shard_dl`——`/root` 属于系统盘,不是挂载的大数据盘(`/root/autodl-tmp`)。仅 openai3-1536 一个数据集的缓存就有9-12GB,系统盘很快见底。

**修法**:把缓存目录挪到 `/root/autodl-tmp` 底下(数据盘,后来又单独扩容到350G)。

## 8. 重跑时把已下好的数据清空又中途被杀,数据丢失

**现象**:openai3-1536 日志明明显示"下载完成、已写盘"(base=999,000 query=1,000),但事后检查 `base.fvecs`/`query.fvecs` 实际是**0字节**。

**原因**:后来为修复另一个问题(见第9条)重新整体跑了一遍,`all` 模式是从头开始处理所有数据集的,重新打开 openai3-1536 的输出文件是 `"wb"` 模式(一打开就清空),这次重跑还没来得及把数据重新写完,就因为发现系统盘写满(第7条)而被 `pkill` 掉了——文件被清空了但没重新填上。

**教训**:`"wb"` 模式的截断是没有"确认"步骤的,一旦打开就回不去;这种"整体重跑"的设计对已完成的部分不友好,好在原始 parquet 分片缓存没丢,重新转换不需要再走网络。

## 9. 本地改完的代码没推送到 GitHub

**现象**:远程 `git pull` 之后运行 `--prepare-only` 参数,报错 "usage: ... " 看起来完全不认识这个参数,一开始以为是"卡住了"。

**原因**:本地工作目录里已经有一版重写过的脚本(效率更高、带 `--prepare-only`),但从来没有 `git commit` + `push` 过,远程仓库里其实还是旧版本,`git pull` 自然拉不到新功能。

**修法**:把本地未提交的版本 `commit` + `push`,远程重新 `pull`。

## 10. curl 走 HTTP/2 被服务器重置连接

**现象**:一次完整跑 4 个数据集的过程,全部以 FAILED 收尾,报错 `curl exit 92`(`CURLE_HTTP2_STREAM`)。

**原因**:`hf-mirror.com` 在 HTTP/2 协议层偶尔把连接流重置掉,不是数据内容问题。

**修法**:curl 命令加 `--http1.1`,强制走 HTTP/1.1,不走 HTTP/2 协商。

## 11. curl 自带的 `--retry` 覆盖不到"传输中途被断开"

**现象**:stella-trec24 已经下到57%(42个分片全部成功),第43个分片下载中途被服务器断开连接(`curl exit 18`,`CURLE_PARTIAL_FILE`),整个数据集直接判定失败,前面57%的进度虽然缓存还在但没能自动扛过这一次。

**原因**:curl 自带的 `--retry` 只覆盖连接超时、HTTP 5xx 这类特定的"瞬时错误",不包括"数据传到一半连接被关闭"这种情况,单个分片抽一次风,`subprocess.run(..., check=True)` 就直接抛异常,拖垮整个数据集。

**修法**:在 Python 这一层再包一层重试(3次),每次重试因为 curl 的 `-C -` 会从上次断点续传,不是冷启动重来,代价很小。

## 小结:根本原因

不是单一的大问题,是好几个独立的坑叠在一起:
1. **这台机器到国外网络本身就不稳定**(国内服务器,是环境问题,不是脚本问题)。
2. **这几个数据集规模是真大**(最大1780万条、74个分片),同样的低概率网络故障,分片数一多,撞上的概率就被放大了很多。
3. **好几个坑只有实际跑一遍才能发现**:`aiohttp` 不认代理、`requests` 静默卡死、数据集自己的 `config`/`split` 命名不符合直觉——这些没法靠读文档或者小规模测试提前发现。
4. 中间也有本可避免的失误:改完代码忘记推送、缓存目录一开始就该放数据盘。
