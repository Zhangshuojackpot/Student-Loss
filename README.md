# Student Loss
This is the official PyTorch implementation of our work [Student Loss: Towards the Probability Assumption in Inaccurate Supervision](https://ieeexplore.ieee.org/abstract/document/10412669), which has been published in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). This repo contains some key codes of our method and its application in MNIST, CIFAR10, and CIFAR100 datasets.<br>
<div align=center>
<img width="800" src="https://github.com/Zhangshuojackpot/Student-Loss/blob/main/introduction.png"/>
</div>

### Abstract
Noisy labels are often encountered in datasets, but learning with them is challenging. Although natural discrepancies between clean and mislabeled samples in a noisy category exist, most techniques in this field still gather them indiscriminately, which leads to their performances being partially robust. In this paper, we reveal both empirically and theoretically that the learning robustness can be improved by assuming deep features with the same labels follow a student distribution, resulting in a more intuitive method called student loss. By embedding the student distribution and exploiting the sharpness of its curve, our method is naturally data-selective and can offer extra strength to resist mislabeled samples. This ability makes clean samples aggregate tightly in the center, while mislabeled samples scatter, even if they share the same label. Additionally, we employ the metric learning strategy and develop a large-margin student (LT) loss for better capability. It should be noted that our approach is the first work that adopts the prior probability assumption in feature representation to decrease the contributions of mislabeled samples. This strategy can enhance various losses to join the student loss family, even if they have been robust losses. Experiments demonstrate that our approach is more effective in inaccurate supervision. Enhanced LT losses significantly outperform various state-of-the-art methods in most cases. Even huge improvements of over 50% can be obtained under some conditions.

### Preparation
The experimental environment is in [requirements.txt](https://github.com/Zhangshuojackpot/Student-Loss/blob/main/requirements.txt).<br>

### Usage
Run [main_lt.py](https://github.com/Zhangshuojackpot/Student-Loss/blob/main/codes_upload_real/main_lt.py) to obtain the results. For example, if you want to obtain the result of the LT-GCE loss under the noise rate of 0.2 of the symmetric noise on MNIST, you can type:<br>
```
python main_lt.py --dataset 'MNIST' --noise_type 'symmetric' --noise_rate 0.2 --is_student 1 --loss 'GCE'
```

### Citation
If you think this repo is useful in your research, please consider citing our paper.
```
@ARTICLE{10412669,
  author={Zhang, Shuo and Li, Jian-Qing and Fujita, Hamido and Li, Yu-Wen and Wang, Deng-Bao and Zhu, Ting-Ting and Zhang, Min-Ling and Liu, Cheng-Yu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Student Loss: Towards the Probability Assumption in Inaccurate Supervision}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TPAMI.2024.3357518}}
```
Meanwhile, our implementation uses parts of some public codes in [Learning With Noisy Labels via Sparse Regularization
](https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_Learning_With_Noisy_Labels_via_Sparse_Regularization_ICCV_2021_paper.html). Please consider citing this paper.
```
@InProceedings{Zhou_2021_ICCV,
    author    = {Zhou, Xiong and Liu, Xianming and Wang, Chenyang and Zhai, Deming and Jiang, Junjun and Ji, Xiangyang},
    title     = {Learning With Noisy Labels via Sparse Regularization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {72-81}
}
```
