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
|YOLOv5n|4w8f|24.9%|42.0%|56.7%|39.3%|wi bias correction|
|YOLOv5n|4w8f|21.1%|36.5%|57.6%|32.0%|wi bias correction & symmetric|
|YOLOv5n|4w8f|25.0%|42.7%|57.6%|39.5%|前6个module(4层conv，2个C3 block)8w|
|YOLOv5n|4w8f|26.1%|43.9%|58.1%|40.8%|前6个module(4层conv，2个C3 block)8w & bc|
|YOLOv5n|4w8f|23.4%|40.4%|58.8%|35.6%|前6个module(4层conv，2个C3 block)8w & bc & symmetric|
||
|YOLOv5s|float|37.1%|56.6%|66.8%|52.1%|
|YOLOv5s|4w8f|33.4%|52.3%|64.5%|47.7%|
|YOLOv5s|4w8f|32.8%|51.8%|64.2%|47.8%|percentile a=0.001|
|YOLOv5s|4w8f|35.3%|54.8%|65.8%|50.0%|wi bias correction|
|YOLOv5s|4w8f|31.8%|50.5%|64.5%|45.1%|wi bias correction & symmetric|
|YOLOv5s|4w8f|35.7%|55.3%|65.8%|50.5%|前6个module(4层conv，2个C3 block)8w|
|YOLOv5s|4w8f|36.0%|55.6%|66.2%|50.6%|前6个module(4层conv，2个C3 block)8w & bc|
|YOLOv5s|4w8f|33.6%|53.2%|65.9%|47.8%|前6个module(4层conv，2个C3 block)8w & bc & symmetric|

### 4w4f
- Weight bit: 4
- Feature bit: 4
- With Bias Correction
- Weight
  - Granularity: group-wise
  - Groupsize: 8
  - Scheme: asymmetric
  - Observer: MSE
- Feature
  - Granularity: tensor-wise
  - Scheme: asymmetric
  - Observer: MSE
- specical
  - first conv: model_0_conv use 8bit
  - output conv: model_24/25/26 use 8bit

|Model|qconfig|mAP50-95|mAP50|prec|recall|remark|
|-----|-----|-----|-----|-----|-----|-----|
|YOLOv5n|float|27.7%|45.6%|57.5%|43.2%|
|YOLOv5n|4w4f|0.24%|0.40%|0.75%|0.27%||
|YOLOv5n|4w4f|12.8%|23.3%|58.9%|17.7%|F: 5bit|
|YOLOv5n|4w4f|19.9%|35.2%|54.9%|32.4%|F: pwlq|
|YOLOv5n|4w4f|16.5%|29.7%|59.4%|24.1%|W: symmetric; F: pwlq|
|YOLOv5n|4w4f|20.1%|35.2%|55.4%|32.4%|F: pwlq & gs=8|
|YOLOv5n|4w4f|23.7%|41.2%|56.9%|38.2%|F: pwlq; 前6module:8bit|
|YOLOv5n|4w4f|24.8%|42.2%|57.2%|39.3%|W: gs=1; F: pwlq & channelwise; 前6module:8bit|
|YOLOv5n|4w4f|21.4%|36.8%|67.3%|26.5%|W: gs=1 & symmetric; F: pwlq & channelwise; 前6module:8bit|
||
|YOLOv5s|float|37.1%|56.6%|66.8%|52.1%|
|YOLOv5s|4w4f|0.00%|0.00%|0.00%|0.16%||
|YOLOv5s|4w4f|8.25%|14.8%|61.3%|10.1%|F: 5bit|
|YOLOv5s|4w4f|31.5%|50.1%|62.6%|45.5%|F: pwlq|
|YOLOv5s|4w4f|28.4%|45.9%|63.7%|39.7%|W: symmetric; F: pwlq|
|YOLOv5s|4w4f|31.6%|50.1%|62.8%|45.4%|F: pwlq & gs=8|
|YOLOv5s|4w4f|33.8%|53.2%|63.3%|48.7%|F: pwlq; 前6module:8bit|
|YOLOv5s|4w4f|34.5%|54.0%|64.9%|49.3%|W: gs=1; F: pwlq & channelwise; 前6module:8bit|
|YOLOv5s|4w4f|31.6%|49.9%|67.3%|41.6%|W: gs=1 & symmetric; F: pwlq & channelwise; 前6module:8bit|