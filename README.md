# HybridSN-HSI-Classification

Pavia University课程复现实验：收录任务三至六，并附原结构3＋1与改动结构2＋2的三个数据集对比。

姓名：**姚博熙**；学号：107552601293；公开仓库链接：<https://github.com/YAO-1-n/HybridSN_work>

实验使用已有本机环境，由助手执行与整理。报告与代码来源如实保留；不将助手执行写成本人独立完成全部操作。

## 按任务查看

| 任务  | 查看入口                                                                                                                                   | 内容             |
| --- | -------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| 任务3 | [03\_任务三\_完整分类实验](03_%E4%BB%BB%E5%8A%A1%E4%B8%89_%E5%AE%8C%E6%95%B4%E5%88%86%E7%B1%BB%E5%AE%9E%E9%AA%8C/README.md)                     | 完整代码、训练日志、评价结果 |
| 任务4 | [04\_任务四\_分类图与错误分析](04_%E4%BB%BB%E5%8A%A1%E5%9B%9B_%E5%88%86%E7%B1%BB%E5%9B%BE%E4%B8%8E%E9%94%99%E8%AF%AF%E5%88%86%E6%9E%90/README.md) | 分类图及至少300字错误分析 |
| 任务5 | [05\_任务五\_三组对比实验](05_%E4%BB%BB%E5%8A%A1%E4%BA%94_%E4%B8%89%E7%BB%84%E5%AF%B9%E6%AF%94%E5%AE%9E%E9%AA%8C/README.md)                     | 三组学习率比较及分析     |
| 任务6 | [06\_任务六\_总结与报告](06_%E4%BB%BB%E5%8A%A1%E5%85%AD_%E6%80%BB%E7%BB%93%E4%B8%8E%E6%8A%A5%E5%91%8A/README.md)                               | 报告Word与八道反思题    |

所有实际文件都在本仓库内部，任务间使用相对链接。代码、结果和权重集中在任务三，避免重复副本。[本地上传清单](00_%E4%B8%8A%E4%BC%A0%E6%B8%85%E5%8D%95.html)可下载后用浏览器打开；GitHub网页以本README和各任务README为查看入口。

## 来源与贡献边界

- 论文：Roy et al., IEEE GRSL 17(2), 277–281, 2020，DOI：[10.1109/LGRS.2019.2918719](https://doi.org/10.1109/LGRS.2019.2918719)。
- [官方Keras代码](https://github.com/gokriznastic/HybridSN)；[PyTorch参考实现](https://github.com/Pancakerr/HybridSN)。
- 架构从已有本地Pavia实现整理；课程工作包括划分、训练中心PCA、三组控制、日志、全图与错误分析、重载核查。不是声称提出HybridSN。
- 原代码MIT说明保留在[LICENSE-original.txt](LICENSE-original.txt)，其适用范围不自动扩展到全部新增报告与材料。

## 数据与运行

不上传原始MAT。到[公开数据来源](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)获取Pavia University影像及GT，按照[数据放置说明](03_%E4%BB%BB%E5%8A%A1%E4%B8%89_%E5%AE%8C%E6%95%B4%E5%88%86%E7%B1%BB%E5%AE%9E%E9%AA%8C/%E5%AE%9E%E9%AA%8C%E9%A1%B9%E7%9B%AE/data/README.md)放入任务三实验项目的data目录。现有数据最初获取路线未核实；本次没有下载或重新许可数据。

环境：Python 3.10.21，PyTorch 2.11.0+cu128，NumPy 2.2.6、SciPy 1.15.3、scikit-learn 1.7.2、Matplotlib 3.10.9；原实验CUDA 12.8，RTX 5070 Ti Laptop GPU。

[requirements.txt](requirements.txt)记录直接依赖。torch的+cu128是对应CUDA构建，不承诺默认PyPI命令能取得；按[PyTorch官方安装入口](https://pytorch.org/get-started/locally/)选择适配机器的安装来源，再安装其余固定依赖。其他环境版本需要自行验证，不能把本次检查视为新机器安装成功。

在仓库根目录、已准备好的环境中，先检查命令和路径：

```text
python check_submission.py
python "03_任务三_完整分类实验/实验项目/run_experiments.py" --help
```

只有主动执行下列命令才会训练或重新推理：

```text
python "03_任务三_完整分类实验/实验项目/run_experiments.py" --smoke
python "03_任务三_完整分类实验/实验项目/run_experiments.py" --epochs 100
python "03_任务三_完整分类实验/实验项目/predict_saved.py"
```

--smoke为两轮流程检查，不是正式成绩。正式训练会重写任务三实验项目内的同名results文件，重跑前另存仓库副本。predict_saved.py需要数据、已保存权重/PCA/划分/预测，并会重写全图及重载记录。无需使用本机绝对路径。

## 实验设置与结果

Pavia：610×340×103，9类，有标签42,776个；训练12,832、验证4,277、测试25,667，分层随机seed666。GT=0不评分。PCA仅拟合训练中心，15维、白化、full solver；25×25 patch、补0；Adam，batch256，weight_decay1e-6，Dropout0.4，各100轮。

每5轮源验证；最高验证OA选择权重，平分取最早。测试用于选模后评价，任务四主模型预先固定lr0.001。三组划分和初始权重摘要相同，只改变学习率。

| 学习率    |    OA % |    AA % |    Kappa | 最佳轮次 |
| ------ | ------: | ------: | -------: | ---: |
| 0.0001 | 99.9377 | 99.8778 | 0.999174 |   40 |
| 0.001  | 99.9844 | 99.9768 | 0.999793 |   55 |
| 0.01   | 43.5968 | 11.1111 | 0.000000 |    5 |

原始数值见[对比CSV](03_%E4%BB%BB%E5%8A%A1%E4%B8%89_%E5%AE%8C%E6%95%B4%E5%88%86%E7%B1%BB%E5%AE%9E%E9%AA%8C/%E5%AE%9E%E9%AA%8C%E9%A1%B9%E7%9B%AE/results/comparison.csv)和各组results.json/per_class.json。0.01完整训练但分类塌缩到多数类，保留失败结果。AA是九类recall平均；JSON中OA/AA用0–1，表中用百分数。

## 限制与报告

随机训练/验证/测试中心互不重叠，但25×25邻域可能重叠；本成绩不代表空间隔离、跨区域泛化，也不能直接当作论文30/70协议复现分数。GT=0预测未验证；三组只有一个种子，不能声称统计显著。未加入后续跨域研究成绩。

[完整实验报告Word](06_%E4%BB%BB%E5%8A%A1%E5%85%AD_%E6%80%BB%E7%BB%93%E4%B8%8E%E6%8A%A5%E5%91%8A/experiment_report.docx) · [八道反思题](06_%E4%BB%BB%E5%8A%A1%E5%85%AD_%E6%80%BB%E7%BB%93%E4%B8%8E%E6%8A%A5%E5%91%8A/%E5%85%AB%E9%81%93%E5%8F%8D%E6%80%9D%E9%A2%98.md) · [上传前检查与材料范围](05_%E4%BB%BB%E5%8A%A1%E4%BA%94_%E4%B8%89%E7%BB%84%E5%AF%B9%E6%AF%94%E5%AE%9E%E9%AA%8C/%E8%A1%A5%E5%85%85%E5%AF%B9%E6%AF%94_3%EF%BC%8B1%E4%B8%8E2%EF%BC%8B2/README.md)

本仓库收录已有实验材料；未在新环境重新训练或推理。

## 新增两结构对比

[三个数据集：3＋1与2＋2全部已有对比](05_%E4%BB%BB%E5%8A%A1%E4%BA%94_%E4%B8%89%E7%BB%84%E5%AF%B9%E6%AF%94%E5%AE%9E%E9%AA%8C/%E8%A1%A5%E5%85%85%E5%AF%B9%E6%AF%94_3%EF%BC%8B1%E4%B8%8E2%EF%BC%8B2/README.md)。各10%/30%训练比例共12组；一个可运行对比脚本、3张CSV表、6张配对分类图、12张混淆矩阵和12张训练曲线，补充约7.7MB；不再包含大权重、断点和预测数组。全部为旧实验，未新训练。

网页上传请分批，每批最多100文件，保持目录结构；具体见补充对比说明。

## 当前上传目录范围

当前收录任务三至六与两结构对比补充。环境配置、依赖、数据来源和运行命令保留在本README与requirements.txt中；目录入口仅列出实际包含的材料。
