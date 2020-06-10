# Authors
* Sizhe Chen, csz729020210@sjtu.edu.cn
* Xiaolin Huang*, xiaolinhuang@sjtu.edu.cn
* Zhengbao He, lstefanie@sjtu.edu.cn
* Chengjin Sun, sunchengjin@sjtu.edu.cn
* Institute of Image Processing and Pattern Recognition in Shanghai Jiao Tong University

# Description
* Details of DAmageNet can be viewed in paper [Universal Adversarial Attack on Attention and the Resulting Dataset DAmageNet](https://arxiv.org/abs/2001.06325)
* DAmageNet is a massive dataset containing universal adversarial samples generated from ImageNet.
* DAmageNet contains 50000 224*224 images, whose original images have been centrally cropped and resized.
* DAmageNet images have an average root mean square deviation of around 7.32 from original samples.
* DAmageNet can fool pretrained models in ImageNet to have error rate up to 85%.
* DAmageNet can fool adversariral-trained models in ImageNet to have error rate up to 70%.

# Data
* [Download](http://www.pami.sjtu.edu.cn/Show/56/122)
* Each file in DAmageNet has the same name as in ILSVRC2012_img_val.
* Test in DAmageNet can be done by test.py
* Unzip DAmageNet and run
```
python test.py DAmageNet VGG19,ResNet50,DenseNet121 0
```

# Demo
<img src="https://github.com/AllenChen1998/DAmageNet/blob/master/intro.png" height="280"><img src="https://github.com/AllenChen1998/DAmageNet/blob/master/results.png" height="280">
