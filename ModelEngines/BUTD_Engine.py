from Engine import Engine
import pickle
import h5py
import os
import torch
import tqdm
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from Utils import clip_gradient,visualize_att,get_sample_image_info,visualize_att_bboxes
from cider.pyciderevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from cider.pyciderevalcap.ciderD.ciderD import CiderD

#--------------------Models using ResNet to extract image features---------------------#
class BUTDSpatial_Eng(Engine):

    def show_additional_rlt(self,additional,image,caption):
        alphas = additional[0]
        alphas = alphas.squeeze(0).cpu()  #(1,max_len,49)->(max_len,49)
        sta_alpha = torch.zeros(1,alphas.size(1))   #time_step_0 has no alpha value, just for convience of visualization
        alphas = torch.cat((sta_alpha,alphas),dim=0)    #(max_len+1,49)
        alphas = alphas.view(alphas.size(0),self.settings['enc_img_size'],self.settings['enc_img_size'])     #(max_len,14,14)/(max_len,7,7)
        alphas = alphas.detach().numpy()
        caption.insert(0,'<sta>')
        caption.append('<end>')
        visualize_att(image=image,alphas=alphas,caption=caption)

#---------------------Models using pretrained Bottom-up features------------------------#
class BUTDDetection_Eng(Engine):

    def get_bottom_up_features(self):
        trainval_data_file = os.path.join(self.data_dir, 'trainval_36.hdf5')
        h_file = h5py.File(trainval_data_file, 'r')
        self.bottom_up_features_dataset = h_file['image_features']
        self.bottom_up_features_indices = pickle.load(open(os.path.join(self.data_dir, 'trainval_36_indices.pkl'), 'rb'))

    def get_bottom_up_bboxes(self):
        trainval_data_file = os.path.join(self.data_dir, 'trainval_36.hdf5')
        h_file = h5py.File(trainval_data_file, 'r')
        self.bottom_up_bboxes_dataset = h_file['image_bboxes']
        self.bottom_up_features_indices = pickle.load(open(os.path.join(self.data_dir, 'trainval_36_indices.pkl'), 'rb'))

    def get_model_params(self):
        cnn_extractor_params = []
        captioner_params = list(self.model.decoder.parameters())
        return cnn_extractor_params,captioner_params

    def training_epoch(self, dataloader, optimizers, criterion, tqdm_visible=True):
        self.get_bottom_up_features()
        self.model.train()
        if tqdm_visible:
            monitor = tqdm.tqdm(dataloader, desc='Training Process')
        else:
            monitor = dataloader
        for batch_i, (img_ids, imgs, captions, lengths) in enumerate(monitor):
            bottom_up_features_this_batch = []
            for img_id in img_ids:
                bottom_up_features_this_batch.append(
                    torch.from_numpy(self.bottom_up_features_dataset[self.bottom_up_features_indices[img_id]]).float()
                )
            bottom_up_features_this_batch = torch.stack(bottom_up_features_this_batch,dim=0)
            bottom_up_features_this_batch = bottom_up_features_this_batch.to(self.device)
            captions = captions.to(self.device)
            lengths = [cap_len - 1 for cap_len in lengths]
            targets = pack_padded_sequence(input=captions[:, 1:], lengths=lengths, batch_first=True)
            self.model.zero_grad()
            predictions = self.model(bottom_up_features_this_batch, captions, lengths)
            loss = criterion(predictions[0], targets[0])
            loss_npy = loss.cpu().detach().numpy()
            if tqdm_visible:
                monitor.set_postfix(Loss=np.round(loss_npy, decimals=4))
            loss.backward()
            for optimizer in optimizers:
                if optimizer is not None:
                    clip_gradient(optimizer, grad_clip=0.1)
                    optimizer.step()

    def eval_captions_json_generation(self,dataloader,eval_beam_size=-1,tqdm_visible=True):
        self.model.eval()
        self.get_bottom_up_features()
        result = []
        print('Generating captions json for evaluation. Beam Search: %s' % (eval_beam_size!=-1))
        if tqdm_visible:monitor = tqdm.tqdm(dataloader, desc='Generating Process')
        else:monitor = dataloader
        for batch_i, (image_ids, images) in enumerate(monitor):
            bottom_up_features_this_batch = []
            for img_id in image_ids:
                bottom_up_features_this_batch.append(
                    torch.from_numpy(self.bottom_up_features_dataset[self.bottom_up_features_indices[img_id.item()]]).float()
                )
            bottom_up_features_this_batch = torch.stack(bottom_up_features_this_batch,dim=0)
            bottom_up_features_this_batch = bottom_up_features_this_batch.to(self.device)
            with torch.no_grad():
                if eval_beam_size!=-1:
                    generated_captions = self.model.beam_search_sampler(bottom_up_features=bottom_up_features_this_batch, beam_size=eval_beam_size)
                else:
                    generated_captions = self.model.sampler(bottom_up_features=bottom_up_features_this_batch, max_len=20)
            captions = generated_captions.cpu().detach().numpy()
            for image_idx in range(captions.shape[0]):
                sampled_ids = captions[image_idx]
                sampled_caption = []
                for word_id in sampled_ids:
                    word = self.caption_vocab.ix2word[word_id]
                    if word == '<end>':
                        break
                    elif word != '<sta>':
                        sampled_caption.append(word)
                sentence = ' '.join(sampled_caption)
                tmp = {'image_id': int(image_ids[image_idx]), 'caption': sentence}
                result.append(tmp)
        return result

    def test(self,use_scst_model,img_root,img_filename,eval_beam_size=-1):
        self.load_pretrained_model(use_scst_model)
        self.get_bottom_up_features()
        self.get_bottom_up_bboxes()
        self.model.eval()
        img_copy,gts = get_sample_image_info(img_root=img_root,img_filename=img_filename)
        datname,splitname,imgidname = img_filename.split('_')   #'COCO','val2014','0XXX0243832.jpg'
        img_id = imgidname.split('.')[0]
        img_id = int(img_id)
        bottom_up_features_this_img = torch.from_numpy(
            self.bottom_up_features_dataset[self.bottom_up_features_indices[img_id]]
        ).float().unsqueeze(0).to(self.device)
        bottom_up_bboxes_this_img = self.bottom_up_bboxes_dataset[self.bottom_up_features_indices[img_id]]  #numpy.ndarry (36,4)
        caption,additional = self.model.eval_test_image(bottom_up_features=bottom_up_features_this_img,caption_vocab=self.caption_vocab,max_len=20,eval_beam_size=eval_beam_size)
        sentence = ' '.join(caption)
        print('Generated caption:')
        print(sentence)
        if len(gts)>0:
            img_id = list(gts.keys())[0]
            res = [{'image_id':img_id,'caption':sentence}]
            tokenizer = PTBTokenizer(_source='gts')
            _gts = tokenizer.tokenize(gts)
            tokenizer = PTBTokenizer(_source='res')
            _res = tokenizer.tokenize(res)
            ciderD_scorer = CiderD(df='COCO14-val')
            ciderD_score,_ = ciderD_scorer.compute_score(gts=_gts,res=_res)
            print('CIDEr-D :%.3f' % (ciderD_score))
        additional.append(bottom_up_bboxes_this_img)
        self.show_additional_rlt(additional,img_copy,caption)

    def show_additional_rlt(self,additional,image,caption):
        alphas = additional[0]
        alphas = alphas.squeeze(0)  #(1,max_len,36)->(max_len,36)
        alphas = alphas.cpu()
        sta_alpha = torch.zeros(1,alphas.size(1))   #time_step_0 has no alpha value, just for convience of visualization
        alphas = torch.cat((sta_alpha,alphas),dim=0)
        alphas = alphas.detach().numpy()
        bboxes = additional[1]  #(36,4)
        caption.insert(0,'<sta>')
        caption.append('<end>')
        visualize_att_bboxes(image=image,alphas=alphas,bboxes=bboxes,caption=caption)



