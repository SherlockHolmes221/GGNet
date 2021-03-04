Code for our CVPR 2021 paper GGNet

## Getting Started
### Installation
pytorch=0.4.1 torchvision=0.2.1 
 ~~~
 git clone https://github.com/SherlockHolmes221/GGNet.git
 cd GGNet
 pip install -r requirements.txt
 cd src/lib/models/networks/DCNv2
 ./make.sh
 ~~~

## Training and Test
### Dataset Preparation
1. Download [HICO-Det]() datasets. Organize them in `Dataset` folder as follows:

    ~~~
    |-- Dataset/
    |   |-- <dataset name>/
    |       |-- images
    |       |-- annotations
    ~~~
2. Download the pre-processed annotations for HICO-Det from the [[websit]]() and replace the original annotations in `Dataset` folder. The pre-processed annotations including

    ~~~
    |-- anotations/
    |   |-- trainval_hico.json
    |   |-- test_hico.json
    |   |-- corre_hico.npy
    ~~~
   Download the corresponding pre-trained models trained on COCO object detection dataset provided by  [CenterNet](https://github.com/xingyizhou/CenterNet).[Hourglass104](https://drive.google.com/open?id=1-5bT5ZF8bXriJ-wAvOjJFrBLvZV2-mlV)). Put them into the `models` folder.

### Training and Testing
~~~
sh experiments/hico/hoidet_hico_hourglass.sh
sh experiments/vcoco/hoidet_vcoco_hourglass.sh
~~~
### Evalution
~~~

~~~

## Results on HICO-DET and V-COCO

**Our Results on HICO-DET dataset**


|Model| Full (def)| Rare (def)| None-Rare (def)|Full (ko)| Rare (ko)| None-Rare (ko)|FPS|Download|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|dla34 |	||	|	| |	||[model]()|
|hourglass104||	|	|	|	|	||[model]()|

**Our Results on V-COCO dataset**


## Citation
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the url LaTeX package.

~~~

~~~

## Acknowledge
Some of the codes are built upon [PPDM]()