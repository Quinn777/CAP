# Toward Robust Diagnosis: A Contour Attention Preserving Adversarial Defense for COVID-19 Detection

This repo is an official implementation of ["Toward Robust Diagnosis: A Contour Attention Preserving Adversarial Defense for COVID-19 Detection"](https://arxiv.org/abs/2211.16806), which is accepted by AAAI 2023. 



## Abstract

As the COVID-19 pandemic puts pressure on healthcare systems worldwide, the computed tomography image based AI diagnostic system has become a sustainable solution for early diagnosis. However, the model-wise vulnerability under adversarial perturbation hinders its deployment in practical situation. The existing adversarial training strategies are difficult to generalized into medical imaging field challenged by complex medical texture features. To overcome this challenge, we propose a Contour Attention Preserving (CAP) method based on lung cavity edge extraction. The contour prior features are injected to attention layer via a parameter regularization and we optimize the robust empirical risk with hybrid distance metric. We then introduce a new cross-nation CT scan dataset to evaluate the generalization capability of the adversarial robustness under distribution shift. Experimental results indicate that the proposed method achieves state-of-the-art performance in multiple adversarial defense and generalization tasks. 

![image-20221202213940039](http://xiangkun-img.oss-cn-shenzhen.aliyuncs.com/image-20221202213940039.png)

<center>The overall framework of the proposed CAP.</center>



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

- [SARS-COV-2](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)
- [COVID19-C](https://www.kaggle.com/datasets/quinn777/covid19c)
- [MosMed-L](https://www.kaggle.com/datasets/quinn777/mosmedlm)
- [MosMed-M](https://www.kaggle.com/datasets/quinn777/mosmedlm)

![image-20221202214959823](http://xiangkun-img.oss-cn-shenzhen.aliyuncs.com/image-20221202214959823.png)

<center><p>Detail information of the datasets.</p></center

Run the following command for SARS-COV-2 with pretrained [Visformer-tiny](https://drive.google.com/file/d/1n9LwZX8Y2LLKzkVqI-euKDdSeXCn35vB/view?usp=share_link):

```
python main.py --config /configs/sars-visformer-cap.json
```

Run the following command for SARS-COV-2 with pretrained [Deit-tiny](https://drive.google.com/file/d/1DbZ-4R72zzVzAfNmRpY5o_Ic7mg97EZ7/view?usp=sharing):

```
python main.py --config /configs/sars-deit-cap.json
```

Run the following command for MosMed-L with pretrained [Visformer-tiny](https://drive.google.com/file/d/1kf9OmQ8pavyoLi5KuXE2hUN1BN-Fw0rS/view?usp=sharing):

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



## Visualization

<img src="http://xiangkun-img.oss-cn-shenzhen.aliyuncs.com/image-20221202213719329.png" alt="image-20221202213719329" style="zoom: 50%;" />

  <center>Saliency map of three models under different training strategies.</center>



## Cite This Paper

If you find our code helpful for your research, please using the following bibtex to cite our paper:

```
@article{xiang2022toward,
  title={Toward Robust Diagnosis: A Contour Attention Preserving Adversarial Defense for COVID-19 Detection},
  author={Xiang, Kun and Zhang, Xing and She, Jinwen and Liu, Jinpeng and Wang, Haohan and Deng, Shiqi and Jiang, Shancheng},
  journal={arXiv preprint arXiv:2211.16806},
  year={2022}
}
```
