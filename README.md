# [FewSAR](https://arxiv.org/abs/2306.09592)
Make few-shot SAR image classification easy.
Few-shot learning (FSL) is one of the significant and hard problems in the field of image classification. However, in contrast to the rapid development of the visible light dataset, the progress in SAR target image classification is much slower. A key reason for this phenomenon is the lacking of a unified benchmark, which may severely overlooked by the current literature. The researchers of SAR target image classification always report their new results on their own datasets and experimental setup. It leads to inefficiency in result comparison and impedes the further progress of this area. Motivated by this observation, we propose a novel few-shot SAR image classification benchmark (FewSAR) to address this issue. FewSAR consists of an open-source Python code library of 15 classic methods in three categories for few-shot SAR image classification. It provides an accessible and customizable testbed for different few-shot SAR image classification task. To further understanding the performance of different few-shot methods, we establish evaluation protocols and conduct extensive experiments within the benchmark. By analyzing the quantitative results and runtime under the same setting, we observe that the accuracy of metric learning methods can achieve the best results. Meta-learning methods and fine-tuning methods perform poorly when applied to few-shot SAR images, primarily due to the bias of existing datasets. We believe that FewSAR will open up a new avenue for future research and development, on real-world challenges at the intersection of SAR image classification and few-shot deep learning.

[FewSAR: A Few-shot SAR Image Classification Benchmark](https://arxiv.org/abs/2306.09592).
Rui Zhang, Ziqi Wang, Yang Li, Jiabao Wang, Zhiteng Wang. In arXiv 2023.<br>

## Supported Methods
### Fine-tuning based methods
+ [Baseline (ICLR 2019)](https://arxiv.org/abs/1904.04232)
+ [Baseline++ (ICLR 2019)](https://arxiv.org/abs/1904.04232)
+ [RFS (ECCV 2020)](https://arxiv.org/abs/2003.11539)
+ [SKD (arxiv 2020)](https://arxiv.org/abs/2006.09785)
### Meta-learning based methods
+ [MAML (ICML 2017)](https://arxiv.org/abs/1703.03400)
+ [Versa (NeurIPS 2018)](https://openreview.net/forum?id=HkxStoC5F7)
+ [R2D2 (ICLR 2019)](https://arxiv.org/abs/1805.08136)
+ [LEO (ICLR 2019)](https://arxiv.org/abs/1807.05960)
+ [MTL (CVPR 2019)](https://arxiv.org/abs/1812.02391)
+ [ANIL (ICLR 2020)](https://arxiv.org/abs/1909.09157)
### Metric-learning based methods
+ [ProtoNet (NeurIPS 2017)](https://arxiv.org/abs/1703.05175)
+ [RelationNet (CVPR 2018)](https://arxiv.org/abs/1711.06025)
+ [ConvaMNet (AAAI 2019)](https://ojs.aaai.org//index.php/AAAI/article/view/4885)
+ [DN4 (CVPR 2019)](https://arxiv.org/abs/1903.12290)
+ [ATL-Net (IJCAI 2020)](https://www.ijcai.org/proceedings/2020/0100.pdf)
+ [FEAT (CVPR 2020)](http://arxiv.org/abs/1812.03664)




## License
This project is licensed under the MIT License. See LICENSE for more details.

## Acknowledgement
FewSAR is an open source project designed to help few-shot SAR image classification researchers quickly understand the classic methods and code structures. We welcome other contributors to use this framework to implement their own or other impressive methods and add them to FewSAR. This library can only be used for academic research. We welcome any feedback during using FewSAR and will try our best to continually improve the library.

## Citation
If you use this code for your research, please cite following paper.
```
@article{DBLP:journals/corr/abs-2306-09592,
  author       = {Rui Zhang and
                  Ziqi Wang and
                  Yang Li and
                  Jiabao Wang and
                  Zhiteng Wang},
  title        = {FewSAR: {A} Few-shot {SAR} Image Classification Benchmark},
  journal      = {CoRR},
  volume       = {abs/2306.09592},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2306.09592},
  doi          = {10.48550/arXiv.2306.09592},
  eprinttype    = {arXiv},
  eprint       = {2306.09592},
  timestamp    = {Thu, 22 Jun 2023 16:55:52 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2306-09592.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@article{li2021LibFewShot,
  title={LibFewShot: A Comprehensive Library for Few-shot Learning},
  author={Li, Wenbin and Dong, Chuanqi and Tian, Pinzhuo and Qin, Tiexin and Yang, Xuesong and Wang, Ziyi and Huo Jing and Shi, Yinghuan and Wang, Lei and Gao, Yang and Luo, Jiebo},
  journal={arXiv preprint arXiv:2109.04898},
  year={2021}
}
```
