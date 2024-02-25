# Student Loss
This is the official PyTorch implementation of our work [Student Loss: Towards the Probability Assumption in Inaccurate Supervision](https://ieeexplore.ieee.org/abstract/document/10412669), which has been published in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). This repo contains some key codes of our HMW and its application in MNIST, CIFAR10, and CIFAR100 datasets.<br>
<div align=center>
<img width="800" src="https://github.com/Zhangshuojackpot/Student-Loss/blob/main/introduction.png"/>
</div>

### Abstract
Noisy labels are often encountered in datasets, but learning with them is challenging. Although natural discrepancies between clean and mislabeled samples in a noisy category exist, most techniques in this field still gather them indiscriminately, which leads to their performances being partially robust. In this paper, we reveal both empirically and theoretically that the learning robustness can be improved by assuming deep features with the same labels follow a student distribution, resulting in a more intuitive method called student loss. By embedding the student distribution and exploiting the sharpness of its curve, our method is naturally data-selective and can offer extra strength to resist mislabeled samples. This ability makes clean samples aggregate tightly in the center, while mislabeled samples scatter, even if they share the same label. Additionally, we employ the metric learning strategy and develop a large-margin student (LT) loss for better capability. It should be noted that our approach is the first work that adopts the prior probability assumption in feature representation to decrease the contributions of mislabeled samples. This strategy can enhance various losses to join the student loss family, even if they have been robust losses. Experiments demonstrate that our approach is more effective in inaccurate supervision. Enhanced LT losses significantly outperform various state-of-the-art methods in most cases. Even huge improvements of over 50% can be obtained under some conditions.
