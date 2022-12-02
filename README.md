# Toward Robust Diagnosis: A Contour Attention Preserving Adversarial Defense for COVID-19 Detection

This repo is an official implementation of ["Toward Robust Diagnosis: A Contour Attention Preserving Adversarial Defense for COVID-19 Detection"](https://github.com/Quinn777/CAP), which is accepted by AAAI 2023. 



## Abstract

As the COVID-19 pandemic puts pressure on healthcare systems worldwide, the computed tomography image based AI diagnostic system has become a sustainable solution for early diagnosis. However, the model-wise vulnerability under adversarial perturbation hinders its deployment in practical situation. The existing adversarial training strategies are difficult to generalized into medical imaging field challenged by complex medical texture features. To overcome this challenge, we propose a Contour Attention Preserving (CAP) method based on lung cavity edge extraction. The contour prior features are injected to attention layer via a parameter regularization and we optimize the robust empirical risk with hybrid distance metric. We then introduce a new cross-nation CT scan dataset to evaluate the generalization capability of the adversarial robustness under distribution shift. Experimental results indicate that the proposed method achieves state-of-the-art performance in multiple adversarial defense and generalization tasks. 

![image-20221202213940039](http://xiangkun-img.oss-cn-shenzhen.aliyuncs.com/image-20221202213940039.png)

<center>The overall framework of the proposed CAP.</center



## Getting Started

Let's start by cloning the repo:

```
git clone https://github.com/Quinn777/CAP
```

Then, you need to install the required packages including: Python 3.7, CUDA 11.6, Pytorch 1.10 and Timm 0.5.4. To install all these packages, simply run:

```
pip3 install -r requirements.txt
```

Download and extract all the datasets:

- SARS-COV-2
- COVID19-C
- MosMed-L
- MosMed-M

![image-20221202214959823](http://xiangkun-img.oss-cn-shenzhen.aliyuncs.com/image-20221202214959823.png)

<center><p>Detail information of the datasets.</p></center

Run the following command for SARS-COV-2 with Visformer-tiny:

```
python main.py --config /configs/sars-visformer-cap.json
```

Run the following command for SARS-COV-2 with Deit-tiny:

```
python main.py --config /configs/sars-deit-cap.json
```

Run the following command for MosMed-L with Visformer-tiny:

```
python main.py --config /configs/mosmed-visformer-cap.json
```

You can select the training and testing method  by changing the status of parameters "train_method" and "test_method" in **config.json** file. Similarly, changing the status of parameter "epsilon", "num_steps" and "step_size" can control adversarial attack power.

```
# configs/xxx.json
"train_method": "cap",
"test_method": "pgd"
"epsilon": 0.03,
"num_steps": 10,
"step_size": 0.00784,
```



## Model Zoo

Coming soon!



## Visualization

<img src="http://xiangkun-img.oss-cn-shenzhen.aliyuncs.com/image-20221202213719329.png" alt="image-20221202213719329" style="zoom: 50%;" />

<center>Saliency map of three models under different training strategies.</center



## Cite This Paper

If you find our code helpful for your research, please using the following bibtex to cite our paper:

```
@article{xxx,
  title={Toward Robust Diagnosis: A Contour Attention Preserving Adversarial Defense for COVID-19 Detection},
  author={Kun Xiang, Xing Zhang, Jinwen She, Jinpeng Liu, Haohan Wang, Shiqi Deng, Shancheng Jiang},
  journal={arXiv preprint arXiv:2211.16806},
  year={2022}
}
```
