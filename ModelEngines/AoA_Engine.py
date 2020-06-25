from Engine import Engine
import os
import h5py
import pickle
import tqdm
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from Utils import clip_gradient

#--------------------Models using ResNet to extract image features---------------------#
class AoASpatial_Eng(Engine):
    def get_model_params(self):
        cnn_extractor_params = list(filter(lambda p: p.requires_grad, self.model.encoder.feature_extractor.parameters()))
        captioner_params = list(self.model.img_feats_porjection.parameters())+\
                           list(self.model.aoa_refine.parameters())+\
                           list(self.model.decoder.parameters())
        return cnn_extractor_params,captioner_params

#---------------------Models using pretrained Bottom-up features------------------------#
class AoADetection_Eng(Engine):
    def get_bottom_up_features(self):
        trainval_data_file = os.path.join(self.data_dir, 'trainval_36.hdf5')
        h_file = h5py.File(trainval_data_file, 'r')
        self.bottom_up_features_dataset = h_file['image_features']
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