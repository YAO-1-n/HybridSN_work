"""Read-only check of included tasks 3-6 and the compact comparison package."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT/'03_任务三_完整分类实验'/'实验项目'
SUPPLEMENT=ROOT/'05_任务五_三组对比实验'/'补充对比_3＋1与2＋2'
required=[ROOT/'README.md',ROOT/'requirements.txt',PROJECT/'model.py',PROJECT/'run_experiments.py',PROJECT/'predict_saved.py',PROJECT/'results/pca.pkl',PROJECT/'results/split.npz']
for folder in ('03_任务三_完整分类实验','04_任务四_分类图与错误分析','05_任务五_三组对比实验','06_任务六_总结与报告'):
    required.append(ROOT/folder/'README.md')
required.extend((ROOT/'04_任务四_分类图与错误分析'/'结果分析.txt',ROOT/'05_任务五_三组对比实验'/'对比分析.txt',ROOT/'06_任务六_总结与报告'/'experiment_report.pdf',ROOT/'06_任务六_总结与报告'/'八道反思题.md'))
for run in ('lr_0.0001','lr_0.001','lr_0.01'):
    required.extend(PROJECT/'results/runs'/run/name for name in ('best_model.pth','results.json','history.json','per_class.json','confusion_matrix.png','training_curves.png'))
required.extend(SUPPLEMENT/name for name in ('README.md','HybridSN_2＋2_对比实验.py','表格/12组结果汇总.csv','表格/6组成对差异.csv','表格/12组混淆矩阵计数.csv'))
for dataset in ('IP','UP','SA'):
    for ratio in (10,30):
        required.append(SUPPLEMENT/f'配对分类图/{dataset}_train_{ratio}_comparison.png')
        for architecture in ('3d3_2d1','3d2_2d2'):
            for kind in ('混淆矩阵','训练曲线'):
                required.append(SUPPLEMENT/kind/f'{dataset}_train_{ratio}_{architecture}.png')
missing=[str(p.relative_to(ROOT)) for p in required if not p.is_file()]
data_missing=[name for name in ('PaviaU.mat','PaviaU_gt.mat') if not (PROJECT/'data'/name).is_file()]
print(json.dumps({'included_tasks':[3,4,5,6],'supplement':'3+1 and 2+2, IP/UP/SA, 10% and 30%','retained_materials_complete':not missing,'checked_required_files':len(required),'missing':missing,'data_files_to_download':data_missing,'training_executed':False},ensure_ascii=False,indent=2))
raise SystemExit(1 if missing else 0)
