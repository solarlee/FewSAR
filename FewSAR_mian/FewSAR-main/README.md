# [FewSAR](https://arxiv.org/abs/2306.09592)
Make few-shot SAR image classification easy.

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
