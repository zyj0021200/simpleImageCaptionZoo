import os
import json
from COCO_Eval_Utils import coco_eval,coco_eval_specific
from Utils import model_construction,init_SGD_optimizer,init_Adam_optimizer,clip_gradient,get_transform,get_sample_image_info,visualize_att
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import tqdm
import numpy as np
from cider.pyciderevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from cider.pyciderevalcap.ciderD.ciderD import CiderD

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, sample_logprobs, seq, reward):

        sample_logprobs = sample_logprobs.view(-1)  # (batch_size * max_len)
        reward = reward.view(-1)
        # set mask elements for all <end> tokens to 0
        mask = (seq > 0).float()  # (batch_size, max_len)
        # account for the <end> token in the mask. We do this by shifting the mask one timestep ahead
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)

        if not mask.is_contiguous():
            mask = mask.contiguous()

        mask = mask.view(-1)
        output = - sample_logprobs * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

def get_self_critical_reward(gen_result, greedy_res, ground_truth, img_ids, caption_vocab, dataset_name, cider_weight=1):

    # ground_truth is the 5 ground truth captions for a mini-batch, which can be aquired from the preprocess_gd function
    # [[c1, c2, c3, c4, c5], [c1, c2, c3, c4, c5],........]. Note that c is a caption placed in a list
    # len(ground_truth) = batch_size. Already duplicated the ground truth captions in dataloader

    batch_size = gen_result.size(0)
    res = []
    gen_result = gen_result.data.cpu().numpy()  # (batch_size, max_len)
    greedy_res = greedy_res.data.cpu().numpy()  # (batch_size, max_len)

    for image_idx in range(batch_size):
        sampled_ids = gen_result[image_idx]
        for endidx in range(len(sampled_ids)-1,-1,-1):
            if sampled_ids[endidx] != 0:
                break
        sampled_ids = sampled_ids[:endidx+1]
        sampled_caption = []
        for word_id in sampled_ids:
            word = caption_vocab.ix2word[word_id]
            sampled_caption.append(word)
        sentence = ' '.join(sampled_caption)
        res.append({'image_id': img_ids[image_idx], 'caption': [sentence]})

    for image_idx in range(batch_size):
        sampled_ids = greedy_res[image_idx]
        sampled_caption = []
        for word_id in sampled_ids:
            word = caption_vocab.ix2word[word_id]
            if word == '<end>':break
            sampled_caption.append(word)
        sentence = ' '.join(sampled_caption)
        res.append({'image_id': img_ids[image_idx], 'caption': [sentence]})

    CiderD_scorer = CiderD(df='%s-train' % dataset_name)
    _, cider_scores = CiderD_scorer.compute_score(ground_truth, res)

    scores = cider_weight * cider_scores
    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)  # gen_result.shape[1] = max_len
    rewards = torch.from_numpy(rewards).float()

    return rewards

class Engine(object):
    def __init__(self,model_settings_json,dataset_name,caption_vocab,device):
        self.model,self.settings = model_construction(model_settings_json=model_settings_json,caption_vocab=caption_vocab,device=device)
        self.device = device
        self.dataset_name = dataset_name
        self.caption_vocab = caption_vocab
        self.tag = 'Model_' + self.settings['model_type'] + '_Dataset_' + dataset_name
        self.model.to(self.device)

    def load_pretrained_model(self,scst_model=False):
        scst_not_found = False
        if scst_model:
            pretrained_scst_model_path = './CheckPoints/%s/' % self.tag + 'Captioner_scst_cp.pth'
            if os.path.exists(pretrained_scst_model_path):
                self.model.load_state_dict(torch.load(pretrained_scst_model_path))
                print('load pretrained scst weights complete.')
            else:
                print('pretrained scst weights not found, try to load pretrained xe weights.')
                scst_not_found = True
        if not(scst_model) or scst_not_found:
            pretrained_model_path = './CheckPoints/%s/' % self.tag + 'Captioner_cp.pth'
            if os.path.exists(pretrained_model_path):
                self.model.load_state_dict(torch.load(pretrained_model_path))
                print('load pretrained xe weights complete.')
            else:print('model checkpoint not found, training from scratch.')

    def load_score_record(self,scst=False):
        best_cider = 0.0
        scst_score_record_path = './CheckPoints/%s/Captioner_scst_cp_score.json' % (self.tag)
        score_record_path = './CheckPoints/%s/Captioner_cp_score.json' % (self.tag)
        if scst and os.path.exists(scst_score_record_path):
            scst_score_record = json.load(open(scst_score_record_path, 'r'))
            best_cider = scst_score_record['cider']
        if not scst and os.path.exists(score_record_path):
            score_record = json.load(open(score_record_path,'r'))
            best_cider = score_record['cider']
        if best_cider != 0.0:print('best cider record: %.3f, model checkpoints below the score record will not be saved.' % best_cider)
        else: print('best cider record not found.')
        return best_cider

    def get_model_params(self):
        cnn_params = list(filter(lambda p: p.requires_grad, self.model.encoder.feature_extractor.parameters()))
        rnn_params = list(self.model.decoder.parameters())
        return cnn_params,rnn_params

    def init_optimizer(self,optimizer_type,params,learning_rate):
        optimizer = None
        if len(params)>0:
            if optimizer_type == 'Adam':
                optimizer = init_Adam_optimizer(params=params,lr=learning_rate)
            elif optimizer_type == 'SGD':
                optimizer = init_SGD_optimizer(params=params,lr=learning_rate)
        return optimizer
    #------------------------------XELoss training---------------------------------#
    def training(self, num_epochs, train_dataloader, eval_dataloader, eval_caption_path, eval_beam_size=-1,
                 load_pretrained_model=False, overwrite_guarantee=True, cnn_FT_start=False, tqdm_visible=True):
        os.makedirs('./CheckPoints/%s' % self.tag, exist_ok=True)
        if load_pretrained_model:self.load_pretrained_model(scst_model=False)
        else:print('training from scratch')
        if overwrite_guarantee:best_cider_record = self.load_score_record(scst=False)
        else:best_cider_record = 0.0

        self.model.cnn_fine_tune(cnn_FT_start)
        cnn_params,rnn_params = self.get_model_params()
        cnn_optimizer = self.init_optimizer(optimizer_type=self.settings['optimizer'],params=cnn_params,learning_rate=self.settings['cnn_lr'])
        rnn_optimizer = self.init_optimizer(optimizer_type=self.settings['optimizer'],params=rnn_params,learning_rate=self.settings['rnn_lr'])

        criterion = nn.CrossEntropyLoss().to(self.device)
        cider_scores = []
        best_cider = 0.0
        best_epoch = 0
        best_cider_woFT = 0.0
        best_epoch_woFT = 0

        for epoch in range(1, num_epochs + 1):
            print('--------------Start training for Epoch %d, CNN_fine_tune:%s-------------' % (epoch, cnn_FT_start))
            self.training_epoch(dataloader=train_dataloader, optimizers=[cnn_optimizer,rnn_optimizer], criterion=criterion, tqdm_visible=tqdm_visible)
            print('--------------Start evaluating for Epoch %d-----------------' % epoch)
            results = self.eval_captions_json_generation(
                dataloader=eval_dataloader,
                eval_beam_size=eval_beam_size,
                tqdm_visible=tqdm_visible
            )
            cider = coco_eval(results=results, eval_caption_path=eval_caption_path)
            cider_scores.append(cider)
            if cider > best_cider:
                if cider > best_cider_record:
                    torch.save(self.model.state_dict(), './CheckPoints/%s/Captioner_cp.pth' % (self.tag))
                    score_record = {'cider':cider}
                    json.dump(score_record,open('./CheckPoints/%s/Captioner_cp_score.json' % (self.tag),'w'))
                best_cider = cider
                best_epoch = epoch
            if len(cider_scores) >= 4:
                last_4 = cider_scores[-4:]
                last_4_max = max(last_4)
                last_4_min = min(last_4)
                if last_4_max != best_cider or abs(last_4_max - last_4_min) <= 0.01:
                    if cnn_FT_start:
                        print('No improvement with CIDEr in the last 4 epochs...Early stopping triggered.')
                        break
                    else:
                        print('No improvement with CIDEr in the last 4 epochs...CNN fine-tune triggered.')
                        best_cider_woFT = best_cider
                        best_epoch_woFT = best_epoch
                        cnn_FT_start = True
                        self.model.cnn_fine_tune(flag=cnn_FT_start)
                        self.load_pretrained_model(scst_model=False)
                        print('load pretrained model from previous best epoch:%d' % best_epoch_woFT)
                        cnn_params,_ = self.get_model_params()
                        cnn_optimizer = self.init_optimizer(optimizer_type=self.settings['optimizer'],params=cnn_params,learning_rate=self.settings['cnn_lr'])
                        cider_scores = []
                        best_cider = 0.0
                        best_epoch = 0

        print('Model of best epoch #:%d with CIDEr score %.3f w/o cnn fine-tune' % (best_epoch_woFT,best_cider_woFT))
        print('Model of best epoch #:%d with CIDEr score %.3f w/ cnn fine-tune' % (best_epoch,best_cider))

    def training_epoch(self, dataloader, optimizers, criterion, tqdm_visible=True):
        self.model.train()
        if tqdm_visible:
            monitor = tqdm.tqdm(dataloader, desc='Training Process')
        else:
            monitor = dataloader
        for batch_i, (_, imgs, captions, lengths) in enumerate(monitor):
            imgs = imgs.to(self.device)
            captions = captions.to(self.device)
            lengths = [cap_len - 1 for cap_len in lengths]
            targets = pack_padded_sequence(input=captions[:, 1:], lengths=lengths, batch_first=True)
            self.model.zero_grad()
            predictions = self.model(imgs, captions, lengths)
            loss = criterion(predictions[0], targets[0])
            loss_npy = loss.cpu().detach().numpy()
            if tqdm_visible:
                monitor.set_postfix(Loss=np.round(loss_npy, decimals=4))
            loss.backward()
            for optimizer in optimizers:
                if optimizer is not None:
                    clip_gradient(optimizer, grad_clip=0.1)
                    optimizer.step()

    #-------------------------------SCST training-----------------------------------------#
    def SCSTtraining(self, num_epochs, train_dataloader, eval_dataloader, eval_caption_path, eval_beam_size=-1,
                     load_pretrained_scst_model=False, overwrite_guarantee=True, cnn_FT_start=True, tqdm_visible=True):
        print('SCST training needs the model pretrained.')
        self.load_pretrained_model(scst_model=load_pretrained_scst_model)
        if overwrite_guarantee:best_scst_cider_record = self.load_score_record(scst=True)
        else:best_scst_cider_record = 0.0
        self.model.cnn_fine_tune(cnn_FT_start)
        cnn_params,rnn_params = self.get_model_params()
        cnn_optimizer = self.init_optimizer(optimizer_type=self.settings['optimizer'],params=cnn_params,learning_rate=self.settings['scst_cnn_lr'])
        rnn_optimizer = self.init_optimizer(optimizer_type=self.settings['optimizer'],params=rnn_params,learning_rate=self.settings['scst_rnn_lr'])

        criterion = RewardCriterion().to(self.device)
        best_cider = 0.0
        best_epoch = 0

        for epoch in range(1,num_epochs + 1):
            print('--------------Start training for Epoch %d, Training_Stage:SCST-------------' % (epoch))
            self.SCST_training_epoch(dataloader=train_dataloader,optimizers=[cnn_optimizer,rnn_optimizer],criterion=criterion,tqdm_visible=tqdm_visible)
            print('--------------Start evaluating for Epoch %d-----------------' % epoch)
            results = self.eval_captions_json_generation(dataloader=eval_dataloader,eval_beam_size=eval_beam_size,tqdm_visible=tqdm_visible)
            cider = coco_eval(results=results,eval_caption_path=eval_caption_path)
            if cider > best_cider:
                if cider > best_scst_cider_record:         #avoid score decreasing
                    torch.save(self.model.state_dict(), './CheckPoints/%s/Captioner_scst_cp.pth' % (self.tag))
                    score_record = {'cider':cider}
                    json.dump(score_record,open('./CheckPoints/%s/Captioner_scst_cp_score.json' % (self.tag),'w'))
                best_cider = cider
                best_epoch = epoch

        print('Model of best epoch #:%d with CIDEr score %.3f in stage:SCST'
                % (best_epoch,best_cider))

    def SCST_training_epoch(self,dataloader,optimizers,criterion,tqdm_visible=True):
        self.model.train()
        if tqdm_visible:monitor = tqdm.tqdm(dataloader,desc='Training Process')
        else:monitor = dataloader
        for batch_i,(imgids,imgs,img_gts) in enumerate(monitor):
            imgs = imgs.to(self.device)

            self.model.zero_grad()
            self.model.eval()
            with torch.no_grad():
                greedy_res = self.model.sampler(imgs,max_len=20)
            self.model.train()
            seq_gen,seqLogprobs = self.model.sampler_rl(imgs,max_len=20)    #(bsize,max_len)
            rewards = get_self_critical_reward(gen_result=seq_gen,greedy_res=greedy_res,ground_truth=img_gts,
                                               img_ids=imgids,caption_vocab = self.caption_vocab,dataset_name=self.dataset_name)

            loss = criterion(seqLogprobs,seq_gen,rewards.to(self.device))
            loss_npy = loss.cpu().detach().numpy()
            if tqdm_visible:
                monitor.set_postfix(Loss=np.round(loss_npy,decimals=4))
            loss.backward()
            for optimizer in optimizers:
                if optimizer is not None:
                    clip_gradient(optimizer,grad_clip=0.25)
                    optimizer.step()

    def eval_captions_json_generation(self,dataloader,eval_beam_size=-1,tqdm_visible=True):
        self.model.eval()
        result = []
        print('Generating captions json for evaluation. Beam Search: %s' % (eval_beam_size!=-1))
        if tqdm_visible:monitor = tqdm.tqdm(dataloader, desc='Generating Process')
        else:monitor = dataloader
        sample_sent_cnt = 0
        for batch_i, (image_ids, images) in enumerate(monitor):
            images = images.to(self.device)
            with torch.no_grad():
                if eval_beam_size!=-1:
                    generated_captions = self.model.beam_search_sampler(images=images, beam_size=eval_beam_size)
                else:
                    generated_captions = self.model.sampler(images=images, max_len=20)
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
                sample_sent_cnt += 1
                if sample_sent_cnt<30:
                    print(sentence)
                tmp = {'image_id': int(image_ids[image_idx]), 'caption': sentence}
                result.append(tmp)
        return result

    def eval(self,dataset,split,eval_scst,eval_dataloader,eval_caption_path,eval_beam_size=-1,output_statics=False,tqdm_visible=True):
        self.load_pretrained_model(scst_model=eval_scst)
        print('--------------Start evaluating for Dataset %s on %s split-----------------' % (dataset,split))
        results = self.eval_captions_json_generation(dataloader=eval_dataloader, eval_beam_size=eval_beam_size,tqdm_visible=tqdm_visible)
        if output_statics:coco_eval_specific(results=results,eval_caption_path=eval_caption_path)
        else:coco_eval(results=results,eval_caption_path=eval_caption_path)

    def test(self,use_scst_model,img_root,img_filename,eval_beam_size=-1):
        self.load_pretrained_model(use_scst_model)
        self.model.eval()
        img_copy,gts = get_sample_image_info(img_root=img_root,img_filename=img_filename)
        img = get_transform()(img_copy).unsqueeze(0)
        img = img.to(self.device)
        caption,additional = self.model.eval_test_image(image=img,caption_vocab=self.caption_vocab,max_len=20,eval_beam_size=eval_beam_size)
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
            ciderD_scorer = CiderD(df='coco-val')
            ciderD_score,_ = ciderD_scorer.compute_score(gts=_gts,res=_res)
            print('CIDEr-D :%.3f' % (ciderD_score))
        self.show_additional_rlt(additional,img_copy,caption)

    def show_additional_rlt(self,additional,image,caption):
        pass

#-----------------specific model engine------------------#
class NIC_Eng(Engine):
    def get_model_params(self):
        cnn_params = list(filter(lambda p: p.requires_grad, self.model.encoder.feature_extractor.parameters()))
        rnn_params = list(self.model.encoder.img_embedding.parameters()) + \
                     list(self.model.encoder.bn.parameters()) + \
                     list(self.model.decoder.parameters())
        return cnn_params,rnn_params

class BUTDSpatial_Eng(Engine):
    def show_additional_rlt(self,additional,image,caption):
        alphas = additional[0]
        alphas = alphas.squeeze(0)  #(1,max_len,49)->(max_len,49)
        alphas = alphas.view(alphas.size(0),self.settings['enc_img_size'],self.settings['enc_img_size'])     #(max_len,14,14)/(max_len,7,7)
        alphas = alphas.cpu().detach().numpy()
        caption.append('<end>')
        visualize_att(image=image,alphas=alphas,caption=caption)