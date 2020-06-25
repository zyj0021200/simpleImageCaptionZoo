import os
import json
from COCO_Eval_Utils import coco_eval,coco_eval_specific
from Utils import model_construction,init_optimizer,set_lr,clip_gradient,get_transform,get_sample_image_info,visualize_att,RewardCriterion,get_self_critical_reward
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import tqdm
import numpy as np
from cider.pyciderevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from cider.pyciderevalcap.ciderD.ciderD import CiderD

class Engine(object):
    def __init__(self,model_settings_json,dataset_name,caption_vocab,data_dir=None,device='cpu'):
        self.model,self.settings = model_construction(model_settings_json=model_settings_json,caption_vocab=caption_vocab,device=device)
        self.device = device
        self.data_dir = data_dir
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
        cnn_extractor_params = list(filter(lambda p: p.requires_grad, self.model.encoder.feature_extractor.parameters()))
        captioner_params = list(self.model.decoder.parameters())
        return cnn_extractor_params,captioner_params

    #------------------------------XELoss training---------------------------------#
    def training(self, num_epochs, train_dataloader, eval_dataloader, eval_caption_path,
                 optimizer_type, lr_opts, ss_opts, use_preset_settings, eval_beam_size=-1,
                 load_pretrained_model=False, overwrite_guarantee=True, cnn_FT_start=False, tqdm_visible=True):

        os.makedirs('./CheckPoints/%s' % self.tag, exist_ok=True)
        if load_pretrained_model:self.load_pretrained_model(scst_model=False)
        else:print('training from scratch')
        if overwrite_guarantee:best_cider_record = self.load_score_record(scst=False)
        else:best_cider_record = 0.0
        if hasattr(self.model,'cnn_fine_tune'):
            self.model.cnn_fine_tune(cnn_FT_start)
        cnn_extractor_params,captioner_params = self.get_model_params()
        #------------Load preset training settings if exists--------------#
        optim_type = optimizer_type
        lr = lr_opts['learning_rate']
        cnn_FT_lr = lr_opts['cnn_FT_learning_rate']
        if use_preset_settings:
            if self.settings.__contains__('optimizer'):
                optim_type = self.settings['optimizer']
                print('training under preset optimizer_type:%s' % optim_type)
            if self.settings.__contains__('lr'):
                lr = self.settings['lr']
                print('training under preset learning_rate:%.6f' % lr)
            if self.settings.__contains__('cnn_FT_lr'):
                cnn_FT_lr = self.settings['cnn_FT_lr']
                print('training under preset cnn_FT_learning_rate:%.6f' % cnn_FT_lr)
        #-----------------------------------------------------------------#
        cnn_extractor_optimizer = init_optimizer(optimizer_type=optim_type,params=cnn_extractor_params,learning_rate=cnn_FT_lr)
        captioner_optimizer = init_optimizer(optimizer_type=optim_type,params=captioner_params,learning_rate=lr)

        criterion = nn.CrossEntropyLoss().to(self.device)
        cider_scores = []
        best_cider = 0.0
        best_epoch = 0
        best_cider_woFT = 0.0
        best_epoch_woFT = 0

        for epoch in range(1, num_epochs + 1):
            print('----------------------Start training for Epoch %d, CNN_fine_tune:%s---------------------' % (epoch, cnn_FT_start))
            if epoch > lr_opts['lr_dec_start_epoch'] and lr_opts['lr_dec_start_epoch'] >= 0:
                frac = (epoch - lr_opts['lr_dec_start_epoch']) // lr_opts['lr_dec_every']
                decay_factor = lr_opts['lr_dec_rate'] ** frac
                current_lr = lr * decay_factor
            else:
                current_lr = lr
            if cnn_extractor_optimizer is not None:set_lr(cnn_extractor_optimizer,min(cnn_FT_lr,current_lr))
            set_lr(captioner_optimizer, current_lr)  # set the decayed rate
            if epoch > ss_opts['ss_start_epoch'] and ss_opts['ss_start_epoch'] >= 0:
                frac = (epoch - ss_opts['ss_start_epoch']) // ss_opts['ss_inc_every']
                ss_prob = min(ss_opts['ss_inc_prob'] * frac, ss_opts['ss_max_prob'])
                self.model.ss_prob = ss_prob
            else:ss_prob = 0.0
            print('|   current_lr: %.6f   cnn_FT_lr: %.6f   current_scheduled_sampling_prob: %.2f   |'
                  % (current_lr,cnn_FT_lr,ss_prob))
            print('------------------------------------------------------------------------------------------')
            self.training_epoch(dataloader=train_dataloader, optimizers=[cnn_extractor_optimizer,captioner_optimizer], criterion=criterion, tqdm_visible=tqdm_visible)

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
            if len(cider_scores) >= 5:
                last_5 = cider_scores[-4:]
                last_5_max = max(last_5)
                last_5_min = min(last_5)
                if last_5_max != best_cider or abs(last_5_max - last_5_min) <= 0.01:
                    if not hasattr(self.model,'cnn_fine_tune') or cnn_FT_start:
                        print('No improvement with CIDEr in the last 5 epochs...Early stopping triggered.')
                        break
                    else:
                        print('No improvement with CIDEr in the last 5 epochs...CNN fine-tune triggered.')
                        best_cider_woFT = best_cider
                        best_epoch_woFT = best_epoch
                        cnn_FT_start = True
                        self.model.cnn_fine_tune(flag=cnn_FT_start)
                        self.load_pretrained_model(scst_model=False)
                        print('load pretrained model from previous best epoch:%d' % best_epoch_woFT)
                        cnn_extractor_params,_ = self.get_model_params()
                        cnn_extractor_optimizer = init_optimizer(optimizer_type=optim_type,params=cnn_extractor_params,learning_rate=cnn_FT_lr)
                        cider_scores = []

        if hasattr(self.model,'cnn_fine_tune'):
            print('Model of best epoch #:%d with CIDEr score %.3f w/o cnn fine-tune' % (best_epoch_woFT,best_cider_woFT))
            print('Model of best epoch #:%d with CIDEr score %.3f w/ cnn fine-tune' % (best_epoch,best_cider))
        else:
            print('Model of best epoch #:%d with CIDEr score %.3f' % (best_epoch,best_cider))

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
    def SCSTtraining(self, num_epochs, train_dataloader, eval_dataloader, eval_caption_path,
                     optimizer_type, scst_lr, scst_cnn_FT_lr, use_preset_settings, eval_beam_size=-1,
                     load_pretrained_scst_model=False, overwrite_guarantee=True, cnn_FT_start=True, tqdm_visible=True):
        print('SCST training needs the model pretrained.')
        self.load_pretrained_model(scst_model=load_pretrained_scst_model)
        if overwrite_guarantee:best_scst_cider_record = self.load_score_record(scst=True)
        else:best_scst_cider_record = 0.0
        if hasattr(self.model,'cnn_fine_tune'):
            self.model.cnn_fine_tune(cnn_FT_start)
        cnn_extractor_params,captioner_params = self.get_model_params()
        #------------Load preset training settings if exists--------------#
        optim_type = optimizer_type
        lr = scst_lr
        cnn_FT_lr = scst_cnn_FT_lr
        if use_preset_settings:
            if self.settings.__contains__('optimizer'):
                optim_type = self.settings['optimizer']
                print('training under preset optimizer_type:%s' % optim_type)
            if self.settings.__contains__('scst_lr'):
                lr = self.settings['scst_lr']
                print('training under preset scst learning_rate:%.6f' % lr)
            if self.settings.__contains__('scst_cnn_FT_lr'):
                cnn_FT_lr = self.settings['scst_cnn_FT_lr']
                print('training under preset scst cnn_FT_learning_rate:%.6f' % cnn_FT_lr)
        #-----------------------------------------------------------------#
        cnn_extractor_optimizer = init_optimizer(optimizer_type=optim_type,params=cnn_extractor_params,learning_rate=cnn_FT_lr)
        captioner_optimizer = init_optimizer(optimizer_type=optim_type,params=captioner_params,learning_rate=lr)

        criterion = RewardCriterion().to(self.device)
        best_cider = 0.0
        best_epoch = 0

        for epoch in range(1,num_epochs + 1):
            print('--------------Start training for Epoch %d, Training_Stage:SCST--------------' % (epoch))
            print('|                 lr: %.6f        cnn_FT_lr: %.6f                 |'
                  % (lr, cnn_FT_lr))
            print('---------------------------------------------------------------------------')
            self.SCST_training_epoch(dataloader=train_dataloader,optimizers=[cnn_extractor_optimizer,captioner_optimizer],criterion=criterion,tqdm_visible=tqdm_visible)
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
            ciderD_scorer = CiderD(df='COCO14-val')
            ciderD_score,_ = ciderD_scorer.compute_score(gts=_gts,res=_res)
            print('CIDEr-D :%.3f' % (ciderD_score))
        self.show_additional_rlt(additional,img_copy,caption)

    def show_additional_rlt(self,additional,image,caption):
        pass
