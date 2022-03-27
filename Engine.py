import os
import json
from COCO_Eval_Utils import coco_eval,coco_eval_specific
from Utils import model_construction,init_optimizer,set_lr,clip_gradient,get_transform,get_sample_image_info,LabelSmoothingLoss,RewardCriterion,get_self_critical_reward
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import tqdm
import time
import pickle
import numpy as np
from cider.pyciderevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from cider.pyciderevalcap.ciderD.ciderD import CiderD

Models_Using_CNN_Extractor = ['NIC','BUTDSpatial','AoASpatial']

class Engine(object):
    def __init__(self,model_settings_json,dataset_name,caption_vocab,data_dir=None,use_bu='unused',device='cpu'):
        self.model,self.settings = model_construction(model_settings_json=model_settings_json,caption_vocab=caption_vocab,device=device)
        self.device = device
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.use_bu = use_bu
        if use_bu in ['adaptive','fixed']:
            self.bu_feat_data_dir = data_dir + '%s_bu_feat' % (use_bu)
            self.bu_bbox_data_dir = data_dir + '%s_bu_bbox' % (use_bu)
        self.caption_vocab = caption_vocab
        self.tag = 'Model_' + self.settings['model_type'] + '_Dataset_' + dataset_name
        self.model.to(self.device)
        if self.settings['model_type'] in Models_Using_CNN_Extractor:self.cnn_ft_model=1
        else: self.cnn_ft_model=0

    def modify_visual_inputs(self,img_tensors,supp_info_datas=[]):
        '''
        This function can be overwritten according to different models
        :param img_tensors: (bsize,3,224,224) raw image tensors
        :param supp_info_datas: 'bu_feats' and 'bu_mask' for models using pretrained faster-rcnn bottom_up features('fixed' or 'adaptive')
        :return: inputs: visual inputs dict.
        '''
        img_tensors = img_tensors.to(self.device)
        inputs = {'img_tensors':img_tensors}
        return inputs

    def load_from_checkpoint(self,load_scst_model=False,load_best=False):
        cp_root_dir = './CheckPoints/%s/' % (self.tag)
        start_epoch = 1
        cider_his = []
        if load_scst_model:scst_flag = 'scst_'
        else:scst_flag = ''
        best_model_not_found = False
        if load_best:
            model_best_path = os.path.join(cp_root_dir,'best/Captioner_%scp.pth' % scst_flag)
            if os.path.exists(model_best_path):
                self.model.load_state_dict(torch.load(model_best_path))
                print('load pretrained %sbest model_state_dict complete.' % scst_flag)
            else:
                print('pretrained best %smodel_state_dict not found, try to load from recent checkpoint.' % scst_flag)
                best_model_not_found = True
        if not load_best or best_model_not_found:
            cider_his = []
            if os.path.exists(os.path.join(cp_root_dir,'cp/%sstate_histories.json' % scst_flag)):
                state_histories = json.load(open(os.path.join(cp_root_dir,'cp/%sstate_histories.json' % scst_flag),'r'))    #{'cider_his':[10,20,30,...],'cnn_FT_flag':1/0}
                cider_his = state_histories['cider_his']    # [10,20,30,...]
            model_cp_path = os.path.join(cp_root_dir,'cp/Captioner_%scp.pth' % scst_flag)
            if os.path.exists(model_cp_path):
                self.model.load_state_dict(torch.load(model_cp_path))
                print('load pretrained %smodel_state_dict from recent checkpoint complete.' % scst_flag)
            else:
                print('recent checkpoint not found.')
            start_epoch = len(cider_his) + 1
        return cider_his,start_epoch

    def load_history_best_score(self,load_scst_record=False):
        best_dir = './CheckPoints/%s/best/' % (self.tag)
        best_cider = 0.0
        if not load_scst_record and os.path.exists(os.path.join(best_dir,'best_score_record.json')):
            best_cider = json.load(open(os.path.join(best_dir,'best_score_record.json'),'r'))['cider']  #{'cider':100}
        elif load_scst_record and os.path.exists(os.path.join(best_dir,'best_scst_score_record.json')):
            best_cider = json.load(open(os.path.join(best_dir,'best_scst_score_record.json'),'r'))['cider']  #{'cider':100}
        return best_cider

    def save_checkpoint(self,cider_scores,save_scst_model=False):
        if save_scst_model:scst_flag = 'scst_'
        else:scst_flag = ''
        cp_dir = './CheckPoints/%s/cp/' % (self.tag)
        model_cp_path = os.path.join(cp_dir,'Captioner_%scp.pth' % scst_flag)
        torch.save(self.model.state_dict(),model_cp_path)
        state_histories = {'cider_his': cider_scores}
        json.dump(state_histories, open(os.path.join(cp_dir,'%sstate_histories.json' % scst_flag), 'w'))

    #------------------------------XELoss training---------------------------------#
    def training(self, start_from, num_epochs,
                 train_dataloader, eval_dataloader, eval_caption_path,
                 optimizer_type, lm_rate, lr_opts, ss_opts,
                 eval_beam_size=-1, tqdm_visible=True):

        os.makedirs('./CheckPoints/%s/cp/' % self.tag, exist_ok=True)
        os.makedirs('./CheckPoints/%s/best/' % self.tag, exist_ok=True)

        cider_his = []
        start_epoch = 1
        cnn_ft_enable = 0
        cider_history_best = self.load_history_best_score(load_scst_record=False)
        print('history best cider on val split w/o beam search: %.3f' % cider_history_best)
        if start_from == 'checkpoint':
            cider_his,start_epoch = self.load_from_checkpoint(load_scst_model=False,load_best=False)
        else:print('training from scratch')

        lr_dict = {
            'lr': lr_opts['learning_rate'],
            'cnn_ft_lr':lr_opts['cnn_finetune_learning_rate']*self.cnn_ft_model
        }
        #criterion = nn.CrossEntropyLoss().to(self.device)
        criterion = LabelSmoothingLoss(smoothing=lm_rate).to(self.device)
        #----------modify cider_scores list to keep consistent with history
        if len(cider_his)>0:            #which means we have loaded the info. from checkpoint
            cider_scores = cider_his
            best_cider = max(cider_scores)
            best_epoch = cider_scores.index(max(cider_scores))
        else:
            cider_scores = []
            best_cider = 0.0
            best_epoch = 0

        for epoch in range(start_epoch, num_epochs + 1):
            print('----------------------Start training for Epoch %d---------------------' % (epoch))
            if epoch > lr_opts['lr_dec_start_epoch'] and lr_opts['lr_dec_start_epoch'] >= 0:
                frac = (epoch - lr_opts['lr_dec_start_epoch']) // lr_opts['lr_dec_every']
                decay_factor = lr_opts['lr_dec_rate'] ** frac
            else:
                decay_factor = 1
            if epoch > lr_opts['cnn_finetune_start'] and self.cnn_ft_model and not cnn_ft_enable:
                cnn_ft_enable = 1
                self.model.cnn_finetune(flag=True)

            current_lr_list = {'lr':lr_dict['lr']*decay_factor,'cnn_ft_lr':min(lr_dict['cnn_ft_lr'],lr_dict['lr']*decay_factor)*cnn_ft_enable}
            optimizer = init_optimizer(optimizer_type=optimizer_type,
                                       params=self.model.get_param_groups(lr_dict=current_lr_list),
                                       learning_rate=current_lr_list['lr'])

            if epoch > ss_opts['ss_start_epoch'] and ss_opts['ss_start_epoch'] >= 0:
                frac = (epoch - ss_opts['ss_start_epoch']) // ss_opts['ss_inc_every']
                ss_prob = min(ss_opts['ss_inc_prob'] * frac, ss_opts['ss_max_prob'])
                self.model.ss_prob = ss_prob
            else:ss_prob = 0.0
            print('|   current_lr: %.6f   cnn_ft_lr: %.6f   current_scheduled_sampling_prob: %.2f   |'
                  % (current_lr_list['lr'],current_lr_list['cnn_ft_lr'],ss_prob))
            print('------------------------------------------------------------------------------------------')
            self.training_epoch(dataloader=train_dataloader, optimizer=optimizer, criterion=criterion, tqdm_visible=tqdm_visible)
            print('--------------Start evaluating for Epoch %d-----------------' % epoch)
            results = self.eval_captions_json_generation(
                dataloader=eval_dataloader,
                eval_beam_size=eval_beam_size,
                tqdm_visible=tqdm_visible
            )
            cider = coco_eval(results=results, eval_caption_path=eval_caption_path)
            cider_scores.append(cider)
            if cider > best_cider:
                if cider > cider_history_best:
                    torch.save(self.model.state_dict(), './CheckPoints/%s/best/Captioner_cp.pth' % (self.tag))
                    score_record = {'cider':cider}
                    json.dump(score_record,open('./CheckPoints/%s/best/best_score_record.json' % (self.tag),'w'))
                best_cider = cider
                best_epoch = epoch

            self.save_checkpoint(cider_scores,save_scst_model=False)

        print('Model of best epoch #:%d with CIDEr score %.3f' % (best_epoch,best_cider))

    def training_epoch(self, dataloader, optimizer, criterion, tqdm_visible=True):
        self.model.train()
        if tqdm_visible:
            monitor = tqdm.tqdm(dataloader, desc='Training Process')
        else:
            monitor = dataloader
        for batch_i, (img_ids, img_tensors, captions, lengths, supp_info_datas) in enumerate(monitor):
            visual_inputs = self.modify_visual_inputs(img_tensors,supp_info_datas)
            captions = captions.to(self.device)
            lengths = [cap_len - 1 for cap_len in lengths]
            targets = pack_padded_sequence(input=captions[:, 1:], lengths=lengths, batch_first=True)
            self.model.zero_grad()
            predictions = self.model(visual_inputs, captions, lengths)
            loss = criterion(predictions[0], targets[0])
            loss_npy = loss.cpu().detach().numpy()
            if tqdm_visible:
                monitor.set_postfix(Loss=np.round(loss_npy, decimals=4))
            loss.backward()
            clip_gradient(optimizer, grad_clip=0.1)
            optimizer.step()

    #-------------------------------SCST training-----------------------------------------#
    def SCSTtraining(self, scst_num_epochs, train_dataloader, eval_dataloader, eval_caption_path,
                     optimizer_type, scst_lr, scst_cnn_FT_lr, eval_beam_size=-1,
                     start_from='stratch', cnn_FT_start=True, tqdm_visible=True):
        print('SCST training needs the model pretrained.')
        os.makedirs('./CheckPoints/%s/cp/' % self.tag, exist_ok=True)
        os.makedirs('./CheckPoints/%s/best/' % self.tag, exist_ok=True)
        #-----------------------------------------------------------------#
        scst_cider_his = []
        scst_start_epoch = 1
        scst_cider_history_best = self.load_history_best_score(load_scst_record=True)
        print('history best scst_cider on val split w/o beam search: %.3f' % scst_cider_history_best)
        if start_from == 'checkpoint':
            scst_cider_his,scst_start_epoch = self.load_from_checkpoint(load_scst_model=True,load_best=False)
        else:
            print('load pretrained model_state_dict before starting scst training...')
            self.load_from_checkpoint(load_scst_model=False,load_best=True)

        if hasattr(self.model,'cnn_fine_tune'):
            self.model.cnn_fine_tune(cnn_FT_start)

        lr_dict = {
            'lr': scst_lr,
            'cnn_ft_lr':scst_cnn_FT_lr*self.cnn_ft_model
        }
        optimizer = init_optimizer(optimizer_type=optimizer_type,
                                   params=self.model.get_param_groups(lr_dict=lr_dict),
                                   learning_rate=lr_dict['lr'])

        criterion = RewardCriterion().to(self.device)
        if len(scst_cider_his)>0:            #which means we have loaded the info. from checkpoint
            cider_scores = scst_cider_his
            best_cider = max(scst_cider_his)
            best_epoch = scst_cider_his.index(max(scst_cider_his))
        else:
            cider_scores = []
            best_cider = 0.0
            best_epoch = 0

        for epoch in range(scst_start_epoch,scst_num_epochs + 1):
            print('--------------Start training for Epoch %d, Training_Stage:SCST--------------' % (epoch))
            print('|                 lr: %.6f        cnn_FT_lr: %.6f                 |'
                  % (lr_dict['lr'], lr_dict['cnn_ft_lr']))
            print('---------------------------------------------------------------------------')
            self.SCST_training_epoch(dataloader=train_dataloader,optimizer=optimizer,criterion=criterion,tqdm_visible=tqdm_visible)
            print('--------------Start evaluating for Epoch %d-----------------' % epoch)
            results = self.eval_captions_json_generation(dataloader=eval_dataloader,eval_beam_size=eval_beam_size,tqdm_visible=tqdm_visible)
            cider = coco_eval(results=results,eval_caption_path=eval_caption_path)
            cider_scores.append(cider)
            if cider > best_cider:
                if cider > scst_cider_history_best:         #avoid score decreasing
                    torch.save(self.model.state_dict(), './CheckPoints/%s/best/Captioner_scst_cp.pth' % (self.tag))
                    score_record = {'cider':cider}
                    json.dump(score_record,open('./CheckPoints/%s/best/Captioner_scst_cp_score.json' % (self.tag),'w'))
                best_cider = cider
                best_epoch = epoch
            self.save_checkpoint(cider_scores=cider_scores,save_scst_model=True)

        print('Model of best epoch #:%d with CIDEr score %.3f in stage:SCST'
                % (best_epoch,best_cider))

    def SCST_training_epoch(self,dataloader,optimizer,criterion,tqdm_visible=True):
        self.model.train()
        if tqdm_visible:monitor = tqdm.tqdm(dataloader,desc='Training Process')
        else:monitor = dataloader
        for batch_i,(img_ids,img_tensors,img_gts,supp_info_datas) in enumerate(monitor):
            visual_inputs = self.modify_visual_inputs(img_tensors=img_tensors,supp_info_datas=supp_info_datas)
            self.model.zero_grad()
            self.model.eval()
            with torch.no_grad():
                greedy_res = self.model.sampler(visual_inputs,max_len=20)
            self.model.train()
            seq_gen,seqLogprobs = self.model.sampler_rl(visual_inputs,max_len=20)    #(bsize,max_len)
            rewards = get_self_critical_reward(gen_result=seq_gen,greedy_res=greedy_res,ground_truth=img_gts,
                                               img_ids=img_ids,caption_vocab = self.caption_vocab,dataset_name=self.dataset_name)

            loss = criterion(seqLogprobs,seq_gen,rewards.to(self.device))
            loss_npy = loss.cpu().detach().numpy()
            if tqdm_visible:
                monitor.set_postfix(Loss=np.round(loss_npy,decimals=4))
            loss.backward()
            clip_gradient(optimizer,grad_clip=0.25)
            optimizer.step()

    def eval_captions_json_generation(self,dataloader,eval_beam_size=-1,tqdm_visible=True):
        self.model.eval()
        result = []
        print('Generating captions json for evaluation. Beam Search: %s' % (eval_beam_size!=-1))
        if tqdm_visible:monitor = tqdm.tqdm(dataloader, desc='Generating Process')
        else:monitor = dataloader
        for batch_i, (image_ids, img_tensors, supp_info_datas) in enumerate(monitor):
            visual_inputs = self.modify_visual_inputs(img_tensors=img_tensors,supp_info_datas=supp_info_datas)
            with torch.no_grad():
                if eval_beam_size!=-1:
                    generated_captions = self.model.beam_search_sampler(visual_inputs=visual_inputs, beam_size=eval_beam_size)
                else:
                    generated_captions = self.model.sampler(visual_inputs=visual_inputs, max_len=20)
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

    def eval(self,dataset,split,eval_scst,eval_best,eval_dataloader,eval_caption_path,eval_beam_size=-1,output_statics=False,tqdm_visible=True):
        self.load_from_checkpoint(load_scst_model=eval_scst,load_best=eval_best)
        print('--------------Start evaluating for Dataset %s on %s split-----------------' % (dataset,split))
        results = self.eval_captions_json_generation(dataloader=eval_dataloader, eval_beam_size=eval_beam_size,tqdm_visible=tqdm_visible)
        if output_statics:coco_eval_specific(results=results,eval_caption_path=eval_caption_path)
        else:coco_eval(results=results,eval_caption_path=eval_caption_path)

    def test(self,use_scst_model,use_best_model,use_bu_feat,img_root,img_filename,eval_beam_size=-1):
        self.load_from_checkpoint(load_scst_model=use_scst_model,load_best=use_best_model)
        self.model.eval()
        img_copy,gts = get_sample_image_info(img_root=img_root,img_filename=img_filename)
        img = get_transform()(img_copy).unsqueeze(0)
        supp_info_datas = []
        if 'COCO' and '2014' in img_filename:
            datname,splitname,imgidname = img_filename.split('_')   #'COCO','val2014','0XXX0243832.jpg'
            img_id = imgidname.split('.')[0]
            img_id = int(img_id)
            if use_bu_feat in ['fixed','adaptive']:
                bu_feat = np.load(os.path.join(self.data_dir,'%s_bu_feat/%s.npz' % (use_bu_feat,str(img_id))))['feat']
                bu_bbox = np.load(os.path.join(self.data_dir,'%s_bu_bbox/%s.npy' % (use_bu_feat,str(img_id))))
                supp_info_data = {'bu_feat':bu_feat,'bu_bbox':bu_bbox}
                supp_info_datas = [supp_info_data]
        visual_inputs = self.modify_visual_inputs(img_tensors=img,supp_info_datas=supp_info_datas)
        caption,additional_outputs = self.model.eval_test_image(visual_inputs=visual_inputs,caption_vocab=self.caption_vocab,max_len=20,eval_beam_size=eval_beam_size)
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
        self.show_additional_rlt(additional_outputs,visual_inputs,img_copy,caption)

    def show_additional_rlt(self,additional_outputs,visual_inputs,image,caption):
        pass
