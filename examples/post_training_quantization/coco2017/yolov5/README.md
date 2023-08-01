## Command
```
python3 main.py --model_name yolov5n --qconfig_path qconfig.yaml --data_path /PATH/TO/COCO --checkpoint_path checkpoints/yolov5n.pth
python3 main.py --model_name yolov5s --qconfig_path qconfig.yaml --data_path /PATH/TO/COCO --checkpoint_path checkpoints/yolov5s.pth
```

## Dataset
-  Download and prepare coco2017 as described in YOLOv5 repo, which should have the following basic structure:

  ```
  coco
  └── images
    └── train2017
    └── val2017
  └── annotations
  └── labels
  └── train2017.txt
  └── val2017.txt
  ```

## Calibration
-  Random sample image paths for calibration:

  ```
  python3 random_sample_calib.py --data_path /PATH/TO/COCO
  ```

## Pretraind model
- create checkpoints dir:
  ```
  mkdir ./checkpoints
  ```
- Download float checkpoints:
    - [yolov5n](https://drive.google.com/file/d/1pcsVQHoHCZ4N0ZB8E2QfDFzCmKfSCOjz/view?usp=sharing)
    - [yolov5s](https://drive.google.com/file/d/1fsDtQtnmNfMM6n0CpslzTMca7xkiaWhq/view?usp=sharing)

## Update Submodule
```
cd $Sparsebit/examples/post_training_quantization/coco2017/yolov5/
git submodule update --init
```

## Requirements
```
pip install -r yolov5/requirements.txt
```

## COCO Benchmark
- Task: COCO
- Eval data num: 5k
- Calibration data num: 128

### 8w8f
- Weight bit: 8
- Feature bit: 8
- Weight
  - Granularity: channel-wise
  - Scheme: symmetric
  - Observer: MinMax
- Feature
  - Granularity: tensor-wise
  - Scheme: asymmetric
  - Observer: MinMax

|Model|qconfig|mAP50-95|mAP50|prec|recall|
|-----|-----|-----|-----|-----|-----|
|YOLOv5n|float|27.7%|45.6%|57.5%|43.2%|
|YOLOv5n|8w8f|27.3%|45.2%|58.0%|42.8%|
||
|YOLOv5s|float|37.1%|56.6%|66.8%|52.1%|
|YOLOv5s|8w8f|36.7%|56.5%|66.2%|52.1%|

### 4w8f
- Weight bit: 4
- Feature bit: 8
- Weight
  - Granularity: group-wise
  - Groupsize: 8
  - Scheme: asymmetric
  - Observer: MSE
- Feature
  - Granularity: tensor-wise
  - Scheme: asymmetric
  - Observer: MinMax
- specical
  - first conv: model_0_conv weight use 8bit
  - output conv: model_24/25/26 weight use 8bit

|Model|qconfig|mAP50-95|mAP50|prec|recall|remark|
|-----|-----|-----|-----|-----|-----|-----|
|YOLOv5n|float|27.7%|45.6%|57.5%|43.2%|
|YOLOv5n|4w8f|16.5%|29.3%|47.8%|28.2%|
|YOLOv5n|4w8f|18.9%|33.5%|48.6%|32.5%|percentile a=0.001|
|YOLOv5n|4w8f|25.0%|42.7%|57.6%|39.5%|前6个module(4层conv，2个C3 block)8w|
||
|YOLOv5s|float|37.1%|56.6%|66.8%|52.1%|
|YOLOv5s|4w8f|33.4%|52.3%|64.5%|47.7%|
|YOLOv5s|4w8f|32.8%|51.8%|64.2%|47.8%|percentile a=0.001|
|YOLOv5s|4w8f|35.7%|55.3%|65.8%|50.5%|前6个module(4层conv，2个C3 block)8w|
