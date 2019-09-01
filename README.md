# ALPHA_OCR_ATTN

## Installation
Install package
```bash
pip install git+ssh://git@github.com/phvan2312/ALPHA_OCR_ATTN.git@robert
```

## Usage
```python
from alpha_ocr.ocr_infer import OCRInferenceModel
saved_model_fn = "where_is_your_saved_model_path"

ocr_model = OCRInferenceModel(saved_model=saved_model_fn, mode='cuda', Prediction='CTC')

# testing with specific image path
image_fn = "where_is_your_image_path"

import cv2
results  = ocr_model.predict(cv2.imread(image_fn))

print (results)

```
### Arguments
* `--saved_model_fn`: weight path.
* `--mode`: 'cuda' or 'cpu', if no cuda found, set to cpu automatically.
* `--Transformation`: select Transformation module [None | TPS].
* `--FeatureExtraction`: select FeatureExtraction module [VGG | RCNN | ResNet].
* `--SequenceModeling`: select SequenceModeling module [None | BiLSTM].
* `--Prediction`: select Prediction module [CTC | Attn].

### Pretrained weights
* `For Attention `: [weight path here](https://drive.google.com/open?id=1mJZT6NCuLmN-GFXWw9CtwODy2RS1hhlx)
* `For CTC Prediction` : [weight path here](https://drive.google.com/open?id=1-bcQ5wF8jR4WFf2_gkwuVChGkM9Ge53l)
