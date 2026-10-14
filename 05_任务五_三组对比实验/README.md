# 任务五：三组学习率对比

[对比分析](对比分析.txt) · [对比CSV](../03_%E4%BB%BB%E5%8A%A1%E4%B8%89_%E5%AE%8C%E6%95%B4%E5%88%86%E7%B1%BB%E5%AE%9E%E9%AA%8C/%E5%AE%9E%E9%AA%8C%E9%A1%B9%E7%9B%AE/results/comparison.csv)

| 学习率    |    OA % |    AA % |    Kappa | 最佳轮次 |
| ------ | ------: | ------: | -------: | ---: |
| 0.0001 | 99.9377 | 99.8778 | 0.999174 |   40 |
| 0.001  | 99.9844 | 99.9768 | 0.999793 |   55 |
| 0.01   | 43.5968 | 11.1111 | 0.000000 |    5 |

[lr\_0.0001 详细指标](../03_%E4%BB%BB%E5%8A%A1%E4%B8%89_%E5%AE%8C%E6%95%B4%E5%88%86%E7%B1%BB%E5%AE%9E%E9%AA%8C/%E5%AE%9E%E9%AA%8C%E9%A1%B9%E7%9B%AE/results/runs/lr_0.0001/results.json)

![lr\_0.0001 训练曲线](../03_%E4%BB%BB%E5%8A%A1%E4%B8%89_%E5%AE%8C%E6%95%B4%E5%88%86%E7%B1%BB%E5%AE%9E%E9%AA%8C/%E5%AE%9E%E9%AA%8C%E9%A1%B9%E7%9B%AE/results/runs/lr_0.0001/training_curves.png)

[lr\_0.001 详细指标](../03_%E4%BB%BB%E5%8A%A1%E4%B8%89_%E5%AE%8C%E6%95%B4%E5%88%86%E7%B1%BB%E5%AE%9E%E9%AA%8C/%E5%AE%9E%E9%AA%8C%E9%A1%B9%E7%9B%AE/results/runs/lr_0.001/results.json)

![lr\_0.001 训练曲线](../03_%E4%BB%BB%E5%8A%A1%E4%B8%89_%E5%AE%8C%E6%95%B4%E5%88%86%E7%B1%BB%E5%AE%9E%E9%AA%8C/%E5%AE%9E%E9%AA%8C%E9%A1%B9%E7%9B%AE/results/runs/lr_0.001/training_curves.png)

[lr\_0.01 详细指标](../03_%E4%BB%BB%E5%8A%A1%E4%B8%89_%E5%AE%8C%E6%95%B4%E5%88%86%E7%B1%BB%E5%AE%9E%E9%AA%8C/%E5%AE%9E%E9%AA%8C%E9%A1%B9%E7%9B%AE/results/runs/lr_0.01/results.json)

![lr\_0.01 训练曲线](../03_%E4%BB%BB%E5%8A%A1%E4%B8%89_%E5%AE%8C%E6%95%B4%E5%88%86%E7%B1%BB%E5%AE%9E%E9%AA%8C/%E5%AE%9E%E9%AA%8C%E9%A1%B9%E7%9B%AE/results/runs/lr_0.01/training_curves.png)

三组均100轮、同划分/同初始化，只改变学习率；失败组也保留。[返回入口](../README.md)。

## 补充结构实验

[原结构3＋1与2＋2：三个数据集、两种训练比例精简资料](补充对比_3＋1与2＋2/README.md)。与上面的三学习率课程结果分开列出，保持协议对应。
