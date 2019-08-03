import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from alpha_ocr.utils import CTCLabelConverter, AttnLabelConverter
from alpha_ocr.dataset import RawDataset, AlignCollate, SingleFileDataset
from alpha_ocr.model import Model

default_opt = {
    'workers':4,
    'batch_max_length':10,
    'imgH':32,
    'imgW':100,
    'rgb':False,
    'character':"0123456789abcdefghijklmnopqrstuvwxyz",
    'PAD':False,
    'Transformation':'TPS',
    'FeatureExtraction':'VGG',
    'SequenceModeling':"BiLSTM",
    'Prediction':'Attn',
    'num_fiducial': 20,
    'input_channel': 1,
    'output_channel': 512,
    'hidden_size': 256,
}

class my_opt:
    def __init__(self, opt):
       for k, v in opt.items():
           setattr(self, k, v)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IntTensor = torch.cuda.IntTensor if torch.cuda.is_available() else torch.IntTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class OCRInferenceModel:
    def __init__(self, opt=default_opt, saved_model=''):
        opt.update({'saved_model':saved_model})
        opt = my_opt(opt)

        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)

        if opt.rgb:
            opt.input_channel = 3

        self.model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)

        self.model = torch.nn.DataParallel(self.model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        self.model.load_state_dict(torch.load(opt.saved_model, map_location='cpu'))

        self.model.eval()

        self.opt = opt

    def predict(self, img_fn):
        opt = self.opt
        AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        demo_data = SingleFileDataset([img_fn], opt=opt)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)

        results = []

        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            with torch.no_grad():
                image = image_tensors.to(device)
                # For max length prediction

                length_for_pred = IntTensor([opt.batch_max_length] * batch_size)
                text_for_pred = LongTensor(batch_size, opt.batch_max_length + 1).fill_(0)

            if 'CTC' in opt.Prediction:
                preds = self.model(image, text_for_pred).log_softmax(2)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_str = self.converter.decode(preds_index.data, preds_size.data)

            else:
                preds = self.model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            print('-' * 80)
            print('image_path\tpredicted_labels')
            print('-' * 80)
            for _id, (img_name, pred) in enumerate(zip(image_path_list, preds_str)):
                if 'Attn' in opt.Prediction:
                    pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

                preds_str[_id] = pred

                print(f'{img_name}\t{pred}')

            results += preds_str

        return results

if __name__ == '__main__':
    saved_model_fn = "/home/vanph/Desktop/alpha/deep-text-recognition-benchmark/saved_models/TPS-VGG-BiLSTM-Attn-Seed1111_saved/best_accuracy.pth"
    ocr_model = OCRInferenceModel(saved_model=saved_model_fn)

    results = ocr_model.predict(img_fn="/home/vanph/Desktop/alpha/deep-text-recognition-benchmark/data/images_rotated/small_text_415.png")
    print (results)