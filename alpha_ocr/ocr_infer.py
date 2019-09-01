import PIL.Image as Image
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from alpha_ocr.utils import CTCLabelConverter, AttnLabelConverter
from alpha_ocr.dataset import RawDataset, AlignCollate, SingleFileDataset, SingleImageArrayDataset
from alpha_ocr.model import Model

default_opt = {
    'workers':0,
    'batch_max_length':10,
    'imgH':32,
    'imgW':100,
    'rgb':False,
    'character':"0123456789abcdefghijklmnopqrstuvwxyz",
    'PAD':False,
    'Transformation':'TPS',
    'FeatureExtraction':'ResNet',
    'SequenceModeling':"BiLSTM",
    'Prediction':'CTC',
    'num_fiducial': 20,
    'input_channel': 1,
    'output_channel': 512,
    'hidden_size': 256,
}

class my_opt:
    def __init__(self, opt):
       for k, v in opt.items():
           setattr(self, k, v)

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
IntTensor = torch.IntTensor
LongTensor = torch.LongTensor

import torch.nn as nn
class Wrapper(nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

class OCRInferenceModel:
    def __init__(self, opt=default_opt, saved_model='', debug_mode=False, mode='cuda',
                 imgH=32,imgW=100,rgb=False,character="0123456789abcdefghijklmnopqrstuvwxyz",
                 PAD=False,Transformation='TPS',FeatureExtraction='ResNet',SequenceModeling="BiLSTM",
                 Prediction='CTC',num_fiducial=20,input_channel=1,output_channel=512,hidden_size=256):

        for k, v in [('imgH',imgH),('imgW',imgW),('rgb',rgb),('character', character), ('PAD',PAD),
                     ('Transformation',Transformation),('FeatureExtraction',FeatureExtraction),
                     ('SequenceModeling', SequenceModeling), ('Prediction',Prediction),
                     ('num_fiducial',num_fiducial),('input_channel',input_channel),
                     ('output_channel',output_channel),('hidden_size',hidden_size)]:

            opt[k] = v

        opt.update({'saved_model':saved_model})
        opt = my_opt(opt)

        self.debug_mode = debug_mode
        if mode == 'cuda' and not torch.cuda.is_available():
            mode = 'cpu'
            print ("mode:gpu but no cuda found, use cpu instead")

        self.device = torch.device(mode)

        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)

        if opt.rgb:
            opt.input_channel = 3

        self.model = Wrapper(Model(opt, self.device).to(device))
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)

        self.model = self.model.to(self.device)

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        self.model.load_state_dict(torch.load(opt.saved_model, map_location='cpu'))

        self.model.eval()

        self.opt = opt

    def predict_batch_alpha_dummy(self, img_inputs, degree=180):
        rotated_img_inputs = [np.array(Image.fromarray(img_input).rotate(degree)) for img_input in img_inputs]
        all_img_inputs = img_inputs + rotated_img_inputs

        opt = self.opt
        AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        demo_data = SingleImageArrayDataset(all_img_inputs, opt=opt)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=len(all_img_inputs),
            shuffle=False,
            num_workers=self.opt.workers,
            collate_fn=AlignCollate_demo, pin_memory=False)

        results = []

        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            with torch.no_grad():
                image = image_tensors.to(self.device)

                # For max length prediction
                length_for_pred = IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = self.model(image, text_for_pred).log_softmax(2)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_str = self.converter.decode(preds_index.data, preds_size.data)
            else:
                preds = self.model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            if self.debug_mode:
                print('-' * 80)
                print('image_path\tpredicted_labels')
                print('-' * 80)

            for _id, (img_name, pred) in enumerate(zip(image_path_list, preds_str)):
                if 'Attn' in opt.Prediction:
                    pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

                preds_str[_id] = pred

                if self.debug_mode:
                    print(f'{img_name}\t{pred}')

            results += preds_str

        _results = results[:len(img_inputs)]
        _rotated_results = results[len(img_inputs):]

        final_results = [[_results[i], _rotated_results[i]] for i in range(len(img_inputs))]

        return final_results

    def predict_batch(self, img_inputs):
        opt = self.opt
        AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        demo_data = SingleImageArrayDataset(img_inputs, opt=opt)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=len(img_inputs),
            shuffle=False,
            num_workers=self.opt.workers,
            collate_fn=AlignCollate_demo, pin_memory=False)

        results = []

        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            with torch.no_grad():
                image = image_tensors.to(self.device)

                # For max length prediction
                length_for_pred = IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = self.model(image, text_for_pred).log_softmax(2)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_str = self.converter.decode(preds_index.data, preds_size.data)
            else:
                preds = self.model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            if self.debug_mode:
                print('-' * 80)
                print('image_path\tpredicted_labels')
                print('-' * 80)

            for _id, (img_name, pred) in enumerate(zip(image_path_list, preds_str)):
                if 'Attn' in opt.Prediction:
                    pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

                preds_str[_id] = pred

                if self.debug_mode:
                    print(f'{img_name}\t{pred}')

            results += preds_str

        return results

    def predict(self, img_input):
        opt = self.opt
        AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        demo_data = SingleImageArrayDataset([img_input], opt=opt)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=1,
            shuffle=False,
            num_workers=self.opt.workers,
            collate_fn=AlignCollate_demo, pin_memory=False)

        results = []

        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            with torch.no_grad():
                image = image_tensors.to(self.device)
                # For max length prediction

                length_for_pred = IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = self.model(image, text_for_pred).log_softmax(2)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_str = self.converter.decode(preds_index.data, preds_size.data)
            else:
                preds = self.model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            if self.debug_mode:
                print('-' * 80)
                print('image_path\tpredicted_labels')
                print('-' * 80)

            for _id, (img_name, pred) in enumerate(zip(image_path_list, preds_str)):
                if 'Attn' in opt.Prediction:
                    pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

                preds_str[_id] = pred

                if self.debug_mode:
                    print(f'{img_name}\t{pred}')

            results += preds_str

        return results

import time
import os

if __name__ == '__main__':
    #saved_model_fn = "/home/vanph/Desktop/saved_models/TPS-VGG-BiLSTM-Attn-Seed1111/best_accuracy_small_200819.pth"
    saved_model_fn = "/home/vanph/Desktop/alpha/ALPHA_OCR_ATTN/saved_models/robert_0109_ctc.pth"
    ocr_model = OCRInferenceModel(saved_model=saved_model_fn, mode='cuda', Prediction='CTC')

    #time.sleep(5)

    import cv2
    # s_time = time.time()
    # results = ocr_model.predict(img_input=cv2.imread("/home/vanph/Desktop/alpha/deep-text-recognition-benchmark/data/130819/textline/510NF/510NF_P32_IP Camera1_Camera_192.168.1.250_20190620073517_20190620075558_717328.mp4_snapshot_09.41_[2019.08.02_22.51.49].jpg_rotate_270.png"))
    # print (results, time.time() - s_time)
    #
    # exit()

    s_time = time.time()
    input_image_1 = cv2.imread("/home/vanph/Desktop/alpha/deep-text-recognition-benchmark/data/270819/data_debug_frame/0.5482728649083591_higro.png")
    input_image_2 = cv2.imread("/home/vanph/Desktop/alpha/deep-text-recognition-benchmark/data/240819/text_line_out/S20/550SF_S20_IP Camera1_Camera_192.168.1.250_20190620141818_20190620143759_1115927.mp4_snapshot_15.03_[2019.08.03_00.12.44].jpg_rotate_270.png")
    input_image_3 = cv2.imread("/home/vanph/Desktop/text_line_3008/text_line/551GPF_S10/_S10/551GPF_S10_IP Camera1_Camera_192.168.1.250_20190815113015_20190815113454_168640.mp4_snapshot_01.21_[2019.08.19_12.19.35].jpg_0_90.png")
    # import PIL.Image
    # import numpy as np
    # input_image_2 = np.array(PIL.Image.fromarray(input_image_1).rotate(angle=180))
    #
    # PIL.Image.fromarray(input_image_1).show()
    # PIL.Image.fromarray(input_image_2).show()
    #
    # results = ocr_model.predict_batch([input_image_1, input_image_2])

    results = ocr_model.predict_batch_alpha_dummy([input_image_1, input_image_2, input_image_3],degree=180)
    print(results, time.time() - s_time)
    #
    # s_time = time.time()
    # results = ocr_model.predict(img_input=cv2.imread("./demo_images/big_text_1.png"))
    # print(results, time.time() - s_time)
    #
    # s_time = time.time()
    # results = ocr_model.predict(img_input=cv2.imread("./demo_images/big_text_2.png"))
    # print(results, time.time() - s_time)