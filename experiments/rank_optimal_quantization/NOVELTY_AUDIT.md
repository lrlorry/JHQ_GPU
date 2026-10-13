# 新颖性审计（NOVELTY_AUDIT）

## 方法与局限

**检索方式：** 2026-10-13 通过 WebSearch 检索题目、摘要和二手综述，未系统通读全文。下表凡写“已核实”，只表示题目、作者、出处和核心思想已从检索结果确认；具体推导细节未逐行对照原文。

**“未核实”的含义：** 未找到直接来源，或只有间接描述。

**结论的效力：** 本审计不能排除遗漏，**不宣称任何“世界首创”**。

## 相关工作与本研究各候选的关系

| 相关工作 | 核心思想 | 与本研究的重叠 | 核实情况 |
|---|---|---|---|
| **JHQ**（Han, Zhang, Trajcevski, PVLDB 19(7), 2026） | 正交（JL）变换后做笛卡尔 Lloyd–Max primary 码，残差相对 primary 重建；primary 筛选，残差精排 | 本研究的基线。源码核验见 THEORY.md 1 | 论文 bib 与本仓库 `paper_adc2026` 一致 |
| **RaBitQ**（Gao, Long, SIGMOD 2024，arXiv 2405.12497） | 相对 IVF 质心归一化后做随机旋转，每维 1 bit；带误差界的无偏内积估计；每向量存校正因子 | 1-bit 加随机旋转与 JHQ 相同；RaBitQ 相对**质心**编码，即本研究 5.6 节的 list 条件阈值。它的精度依赖每向量浮点校正量，本研究的 bit 预算约束不允许这类量 | 已核实（题目、出处、思想） |
| **Extended RaBitQ / 渐近最优量化**（Gao 等，2024–2025） | 多 bit 扩展，渐近最优误差界 | 不同 bit 预算，不直接比较 | 已核实（检索摘要） |
| **SAQ**（Li 等，PACMMOD 2025，arXiv 2509.12086） | PCA 后按维段分配不同 bit 数，并做编码调整 | “利用维度方差不均”的思路；本研究固定每维 1 bit，不做 bit 分配 | 已核实（检索摘要） |
| **LVQ**（Aguerrebere 等，PVLDB 16(11), 2023） | 先减去数据集均值，再做每向量自适应的标量量化 | 其中“先去均值”与第 2 轮的去中心化阈值同理；LVQ 有每向量缩放量，本研究没有 | 已核实（经 SAQ 论文的二手描述） |
| **ITQ**（Gong, Lazebnik, CVPR 2011 / TPAMI 2013） | 先对数据零中心化，再学习旋转以最小化二值化误差 | 去中心化是 ITQ 的前提步骤；学习旋转不在本研究范围（保留 JL） | 已核实 |
| **Asymmetric distances for binary embeddings**（Gordo, Perronnin, Gong, Lazebnik, CVPR 2011 / TPAMI 2014） | 查询不二值化、库向量二值化的非对称距离，优于 Hamming | 与 JHQ 的 primary（全精度查询 × 1-bit 码）属于同一范式；论文中基于期望的重建与 THEORY.md 2.2 的条件期望距离同类 | 题目与出处已核实；“期望重建”的具体形式**未核实** |
| **Sign-full random projections**（P. Li, AAAI 2019，arXiv 1805.00533） | 一侧全精度、一侧只取符号；E[sgn(x)·y] = √(2/π)·ρ；非线性加权估计量接近 MLE | **与第 1 轮的 LLR 加权最接近**：同样是在“全精度查询 × 符号码”下对相似度做非线性加权的统计估计 | 已核实（检索摘要） |
| **Coding for random projections**（Li, Mitzenmacher, Shrivastava, ICML 2014） | 随机投影的量化编码方案（均匀量化、偏移量、2-bit 非均匀方案） | 与“阈值和偏移的选择”相关；1-bit 偏移阈值的具体结论**未核实** | 部分核实 |
| **Hamming Distance Metric Learning**（Norouzi, Fleet, Salakhutdinov, NIPS 2012）；**Minimal Loss Hashing**（ICML 2011） | 用三元组排序损失学习二值码 | 第 3 轮（排序损失学习阈值）属于这一大类，只是参数化不同 | 已核实 |
| **Supervised Binary Hash Code Learning with JS divergence**（L. Fan, ICCV 2013） | 推导哈希函数的 Bayes 判决误差上界，并以此为目标学习哈希 | 与第 3 节的误排序界思路接近，但用的是 JS 散度而不是 Bhattacharyya | 已核实（检索摘要） |
| **中位数阈值**（如 Self-Taught Hashing，2010；NVIDIA cuVS binary quantizer 支持按维均值或中位数阈值） | 用每维中位数做阈值，使 bit 平衡、熵最大 | **与第 2 轮的 `median` 完全相同**；本研究只是从误排序界推出了它 | 已核实（检索摘要） |
| **Bhattacharyya / Chernoff 界**（经典，Bhattacharyya 1943；Kailath 1967） | 两类判别误差的指数上界 | THEORY.md 4.1 的配对版本是直接应用，**不是新定理** | 经典结果 |
| **概率排序原则**（Robertson 1977） | 按相关概率排序最优 | THEORY.md 2.3 引用 | 经典结果 |
| **Bussgang 定理与 arcsine 律**（Bussgang 1952；Van Vleck） | 高斯输入经符号函数后的相关结构，1-bit 信号的 LMMSE 估计 | 第 4 轮的数学基础 | 经典结果 |
| **Bussgang LMMSE（BLMMSE）一比特 MIMO 信道估计**（Li, Tao, Seco-Granados, Mezghani, Swindlehurst, Liu, IEEE TSP 2017，arXiv 1609.07427） | 用 Bussgang 分解把 1-bit 量化写成线性映射加不相关噪声，arcsine 律给出量化输出的协方差，构造 LMMSE 估计量 | **与第 4 轮的估计量在数学上相同**；本研究只是把它用在 ANN 的码而不是信道上 | 已核实（检索摘要） |
| **Hashing with Binary Autoencoders**（Carreira-Perpiñán, Raziperchikolaei, CVPR 2015） | 线性哈希函数加线性重建映射，用辅助坐标法交替优化 | 二值码的线性解码器同样出现在这里，不过它是与码一起学习的 | 题目、出处、“线性编解码”已核实；具体解码器形式**未核实** |

## 各候选方法的新颖性判断（随迭代更新）

| 候选 | 判断 | 理由 |
|---|---|---|
| 第 1 轮：近邻 LLR 打分 | **已有方法的简单变体** | 数学上等价于对符号内积做奇函数重加权（THEORY.md 3.4）；同类非线性加权估计量见 P. Li（AAAI 2019） |
| 第 2 轮：排序最优阈值，即每维中位数 | **已有做法** | 中位数阈值是哈希中的标准做法。本研究的贡献只是“由误排序界导出，并量化了 JHQ 的零阈值假设在真实 embedding 上的代价”，属于分析，不是新方法 |
| 第 3 轮：成对排序损失学习阈值和电平 | **已有方法的简单变体** | 属于用排序损失学习二值码这一类（Norouzi 等 2011/2012 等），只是参数化为逐维阈值和电平。学到的阈值回到中位数，本身也说明它没有带来新结构 |
| 第 4 轮：线性 MMSE（Bussgang）解码打分 | **已有工具的新应用**（价值待实验确定） | 估计量与一比特 MIMO 的 BLMMSE（Li 等，TSP 2017）数学上相同；二值码的线性解码器见 Binary Autoencoder（CVPR 2015）。可能的贡献只是：用于 JHQ 非对称筛选的查询端变换 w = Bᵀq′（不改码和查表结构），以及“符号噪声白化 + 有效秩低”对 ANN 收益的解释 |
| 第 5 轮 | 见 ITERATION_LOG.md 与最终报告 | — |
