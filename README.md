# Super Resolution
 
This project aims to implement a [Unet-like model](https://arxiv.org/abs/1505.04597) that performs the **super resolution** task of receiving an input image of 64x64 size and outputting an image of the same content with the size increases by 4 times (256x256). The dataset we use can be downloaded [here](https://drive.google.com/file/d/17NiVpVxpkvbc2WDvz1VP0sos6NdbPjLE/view?usp=sharing). 

![loss](https://github.com/quocviethere/unet-super-resolution/assets/96617645/e47669c4-78d4-4fd8-994e-8d8ad0e28ea1)

---
# Implementation

To reproduce our result, simply clone the repository using
```
git clone https://github.com/quocviethere/unet-super-resolution
```

Then run:
```
python main.py
```

By default, when you run the code, it will train the UNet model with [Skip Connection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), to train the model without Skip Connection, you can modify ```main.py``` as follows:

```python
from model import SR_Unet_NoSkip


SR_unet_model, metrics = train_model(
    SR_Unet_NoSkip,
    'SR_Unet_NoSkip',
    save_model,
    optimizer,
    criterion,
    train_loader,
    test_loader,
    EPOCHS,
    device
)
```

The Colab Notebook is available here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BrAz-rsbnTKU4nc4M0_CFjVFkMb4oI30?usp=sharing) 

---

# Model Checkpoints

We provide the model checkpoints for both version:

| **Description**    | **Link**                |
|--------------------|-------------------------|
| w/o Skip Connection | [SR_unet_model_noskip.pt](https://drive.google.com/file/d/11q2N6A7FfEbllrsLxKKqAfdKNQogDlHK/view?usp=sharing) |
| w/ Skip Connecion     | [SR_unet_model.pt](https://drive.google.com/file/d/1evXaXK60835fO1vXxF7mXY3cnMUu5bAI/view?usp=sharing)        |

---

# Results

![results](https://github.com/quocviethere/unet-super-resolution/assets/96617645/8bf2ecf7-8c9f-46be-9491-b34b8f83dea1)

---

# Citations

```bibtex
@INPROCEEDINGS{7780459,
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Deep Residual Learning for Image Recognition}, 
  year={2016},
  volume={},
  number={},
  pages={770-778},
  doi={10.1109/CVPR.2016.90}}
```

```bibtex
@misc{ronneberger2015unet,
      title={U-Net: Convolutional Networks for Biomedical Image Segmentation}, 
      author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
      year={2015},
      eprint={1505.04597},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
