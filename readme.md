  # SPPSFormer
This is the PyTorch implementation of this paper.
> SPPSFormer: High-quality Superpoint-based Transformer for Roof Plane Instance Segmentation from Point Clouds
>
> Cheng Zeng, Xiatian Qi, Huifan Wang, Kai Sun, Pengcheng Zhong, Qiao Xu, Yan Meng, Yangjie Sun and Yuxuan Liu

## Dataset

The real building point clouds are selected from [RoofN3D](https://roofn3d.gis.tu-berlin.de/).

If you download the dataset directly from the RoofN3D official website, please process it following the method described in our paper and place the dataset in the following structure.

```
SPPSFormer-main
├── data
│   ├── rn3d
│   │   ├── train
│   │   ├── val
```

After placing the dataset, run the following commands in order:

```
python data/superpoint_stage1.py
python data/superpoint_stage2.py
python data/prepare_data.py
```

This will generate the superpoint-labeled files used in this paper and the .pth files required for model input.

Alternatively, you can use our pre-processed dataset (including .txt and .pth files), which is available for download at the following Baidu Yun link: [Dataset](https://pan.baidu.com/s/1Sf5QLm9Jz-0y0z7U_JZBUg?pwd=piuu) . After downloading, please place the dataset in the following directory structure:

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

Below is an example of how to create a Conda environment:

```
conda create -n sppsformer python=3.8
conda activate sppsformer
```

Install the dependencies:

```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch
conda install torch-scatter==2.0.9 -c conda-forge
conda install spconv-cu113==2.3.6 cumm-cu113==0.4.11 -c conda-forge

conda install cudatoolkit=11.7.0 cudnn=8.2.1.32 -y

pip install -r requirements.txt
```

Install pointgroup_ops:

```
cd spformer/lib
python setup.py develop
```

## Training

```
python tools/train.py configs/rn3d.yaml
```

The resulting model weights will be stored in the exps folder.

## Testing and visualization

For testing, you can also use our pre-trained weights, which are available for download at the following Baidu Yun link:[Model weights](https://pan.baidu.com/s/1DuPhHMpgtb8qRbry1ApbBQ?pwd=2tha)

```
python tools/test.py configs/rn3d.yaml exps/rn3d/epoch_0100.pth --out out_rn3d
python tools/visualize_X.py
```

Finally, the txt files containing the ground-truth instance labels and predicted instance labels will be saved in the directory specified in the code. For visualization, please use software such as CloudCompare.

## Post-processing

```
python tools/postprocessing_Batch_2.py
python tools/boundaryRefinement_Batch.py
```

## Citation

If you find this work useful in your research, please cite:

