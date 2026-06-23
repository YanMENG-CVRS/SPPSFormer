  # SPPSFormer
这是本篇论文的pytorch实现
> SPPSFormer: High-quality Superpoint-based Transformer for Roof Plane Instance Segmentation from Point Clouds
>
> Cheng Zeng, Xiatian Qi, Huifan Wang, Kai Sun, Pengcheng Zhong, Qiao Xu, Yan Meng, Yangjie Sun and Yuxuan Liu

## Dataset

The real building point clouds are selected from [RoofN3D](https://github.com/sarthakTUM/roofn3d).

如果你是直接从RoofN3D官网下载数据集，请按照我们论文所述的方法进行处理，并按照以下格式放置数据集

```
SPPSFormer-main
├── data
│   ├── rn3d
│   │   ├── train
│   │   ├── val
```

放置好后，依次运行

```
python data/superpoint1.py
python data/superpoint2.py
python data/prepare_data.py
```

即可生成本文所使用的带有超点标签的文件以及模型输入所需要的pth文件

当然，你也可以使用我们已经处理好的数据集（包含txt和pth文件），数据集下载链接：**（记得加百度网盘链接）**，并按照以下格式放置数据集

```
SPPSFormer-main
├── data
│   ├── rn3d
│   │   ├── train
│   │   ├── val
```

## Environment

Requirements

- Python 3.x
- Pytorch 1.10
- CUDA 10.x or higher

下面将会给出创建示例

创建conda环境

```
conda create -n sppsformer python=3.8
conda activate sppsformer
```

安装依赖

```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch
conda install torch-scatter==2.0.9 -c conda-forge
conda install spconv-cu113==2.3.6 cumm-cu113==0.4.11 -c conda-forge

conda install cudatoolkit=11.7.0 cudnn=8.2.1.32 -y

pip install -r requirements.txt
```

安装pointgroup_ops

```
cd spformer/lib
python setup.py develop
```

## Training

```
python tools/train.py configs/rn3d.yaml
```

生成的权重文件会保存在exps目录下面

## Testing and visualization

测试也可以使用我们已经训练好的权重文件，权重文件下载路径：**（记得加百度网盘链接）**

```
python tools/test.py configs/rn3d.yaml exps/rn3d/epoch_0100.pth --out out_rn3d
python tools/visualize_X.py
```

最后带有实例标签和预测实例标签的txt文件保存在代码指定目录下面，若想查看可视化，请使用cloudcompare等软件进行

## Post-processing

```
python tools/postprocessing_Batch_2.py
python tools/boundaryRefinement_Batch.py
```

## Citation

If you find this work useful in your research, please cite:

