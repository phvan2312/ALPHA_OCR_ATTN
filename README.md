# ALPHA_OCR_ATTN

## Installation
Install package
```bash
pip install git+ssh://git@github.com/phvan2312/ALPHA_OCR_ATTN.git
```

## Usage
Please set ENVIRONMENT CUDA_VISIBLE_DEVICES='' to run on CPU. 
```python
from alpha_ocr.ocr_infer import OCRInferenceModel
saved_model_fn = "where_is_your_saved_model_path"

ocr_model = OCRInferenceModel(saved_model=saved_model_fn)

# testing with specific image path
image_fn = "where_is_your_image_path"
results  = ocr_model.predict(image_fn)

print (results)

```

### Pretrained weights
You can download pretrained weights via this link: https://drive.google.com/open?id=1YCtzLvkJcYHaccFNh_LXt4v2i-FpcgpW
