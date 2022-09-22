# RoadSegmentationCRF(Done)
本仓库主要记录遥感图像道路提取任务中后处理的方法，以UNet为例

### 文件介绍
#### 数据集结构
使用massachusetts 或者 DeepGlobel2018都行
```
dataset_*
|	train
		| data
		| label
|	test
		| data
		| label
|	valid
		| data
		| label
```
#### models
- 保存模型的网络结构（UNet）
#### postprocessing
- CRF的实现和参数
#### results
- 保存分割结果
#### weights
- 保存模型的权重
#### evaluate.py
- 保存评价指标
#### predict_crf.py
- 计算用CRF 做后处理之后的mIOU和mDC以及保存结果到results中
#### predict.py
- 计算UNet（不做CRF）的mIOU和mDC以及保存结果到results中
#### roadDataset.py
- 加载数据
### 实验结果
| confg name | mIOU | mDC  |
| :--------: | :--------: | :--------: |
| UNet | 0.6174 | 0.7447 |
| pp_config_p1 |0.6004 |0.7296 |
| pp_config_p2 | 0.6004 | 0.7296 |
| pp_config_p3 | 0.5984 | 0.7281 |
| pp_config_p4 | 0.6027 | 0.7317 |
| pp_config_p5 | 0.5984 | 0.7281 |
| pp_config_p6 | 0.3843 | 0.6208 |
| pp_config_p7 | 0.5984 | 0.7281 |
| pp_config_p8 | 0.6027 | 0.7317 |
| pp_config_p9 | 0.6070 | 0.7355 |

- 实验效果经过CRF 后处理之后，mIOU与mDC都比不经过CRF后处理之后要低。
- 代码以及CRF 参数[参考链接](https://gitlab.com/nicolas-kuechler/cil-road-segmentation-2018/-/tree/master/road-segmentation/postprocessing)
