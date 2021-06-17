# GAN-and-CNN-for-daylight

CNN and GAN are applied to real-time indoor daylight performance prediction.

paper: xxx


## Datasets
- parametric: 600 cases, all shared in **dataset_parametric.rar**
- real-case: 6 cases of 575 are shared (not allowed to share all)

## Models
### ResNet
ResNet is trained on both the parametric box dataset and real-case dataset.

ResNet predicts the overall daylight metrics for a floorplan.
- For static prediction, 3 metrics (mean lux, uniformity and success rate) are predicted.
- For annual prediction, 2 metrics (sDA and UDI) are predicted.

### pix2pix
Pix2pix is trained on the real-case dataset.

### train on your own datasets
To train the models on your own datasets, replace the folder 'datasets' or change datasets path.

## Acknowledgments
Code borrows heavily from [pix2pix](https://github.com/yenchenlin/pix2pix-tensorflow) and [ResNet](https://github.com/calmisential/TensorFlow2.0_ResNet). Thanks for their excellent work!

## License
MIT
