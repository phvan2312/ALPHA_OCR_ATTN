from setuptools import setup

requirements = open('requirements.txt').read().splitlines()

setup(name='ALPHA_OCR_ATTN',
      description='ALPHA OPTICAL CHARACTER RECOGNITION.',
      version='0.1.0',

      packages=['alpha_ocr','alpha_ocr.modules'],
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False)