# ALPHA_OCR_ATTN

## Installation
Install package
```bash
pip install git+ssh://git@github.com/phvan2312/ALPHA_OCR_ATTN.git@fix_gpu_speed_v2
```

## Usage
Please set ENVIRONMENT CUDA_VISIBLE_DEVICES='' to run on CPU. 
```python
from alpha_ocr.ocr_infer import OCRInferenceModel
saved_model_fn = "where_is_your_saved_model_path"

ocr_model = OCRInferenceModel(saved_model=saved_model_fn)

# testing with specific image path
image_fn = "where_is_your_image_path"

import cv2
results  = ocr_model.predict(cv2.imread(image_fn))

print (results)

```

### Pretrained weights
You can download pretrained weights via this link: https://drive.google.com/open?id=1y6n0X8VNHw185y72QGmnu8jQi2u27KlT

### Some examples
| demo images | Prediction |
| ---         |     ---      |
| <img src="./demo_images/big_text_1.png" width="300" height="100">    |   550sf   |
| <img src="./demo_images/big_text_2.png" width="300" height="100">      |    550sf    |
| <img src="./demo_images/small_text_1.png" width="300" height="100">  |   s20   |
| <img src="./demo_images/small_text_2.png" width="300" height="100">  |   p32   |
