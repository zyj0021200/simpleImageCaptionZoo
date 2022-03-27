import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as Fun
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.weight_norm import weight_norm

class EncoderCNN(nn.Module):
    def __init__(self,encoded_img_size=7):
        super(EncoderCNN,self).__init__()
        self.enc_img_size = encoded_img_size
        resnet = models.resnet101(pretrained=True)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_img_size,encoded_img_size))

    def forward(self,images):
        '''
        extract spatial features from raw image tensors.
        :param images: (bsize,3,224,224)
        :return: embed_features: (bsize,49,2048)
        '''
        features = self.feature_extractor(images)   #(bsize,2048,H/32,W/32)
        features = self.adaptive_pool(features)    #(bsize,2048,7,7)
        bsize = images.size(0)
        num_pixels = features.size(2) * features.size(3)
        num_channels = features.size(1)
        features = features.permute(0,2,3,1)    #(bsize,7,7,2048)
        features = features.view(bsize,num_pixels,num_channels)     #(bsize,49,2048)
        return features

class SoftAttention(nn.Module):
    def __init__(self,enc_dim,hidden_dim,atten_dim,dropout=0.5):
        super(SoftAttention,self).__init__()
        self.enc_att = weight_norm(nn.Linear(enc_dim,atten_dim))
        self.dec_att = weight_norm(nn.Linear(hidden_dim,atten_dim))
        self.affine = weight_norm(nn.Linear(atten_dim,1))
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self,enc_features,dec_hidden):
        '''
        typical concat attention
        :param enc_features:(bsize,num_pixels,enc_feature_dim)
        :param dec_hidden:(bsize,hidden_dim)
        :return:atten_weighted_enc:(bsize,enc_feature_dim)
                alpha:(bsize,num_pixels) attention weight over different pixels at each time_step
        '''
        enc_ctx = self.enc_att(enc_features)    #(bsize_t,num_pixels,atten_dim)
        dec_ctx = self.dec_att(dec_hidden)      #(bsize_t,atten_dim)
        atten = self.affine(self.dropout(self.relu(enc_ctx + dec_ctx.unsqueeze(1)))).squeeze(2)  #(bsize_t,num_pixels)
        alpha = torch.softmax(atten,dim=1)
        atten_weighted_enc = (enc_features * alpha.unsqueeze(2)).sum(dim=1)  #(bsize_t,num_pixels,2048)->(bsize_t,2048)
        return atten_weighted_enc,alpha

class DecoderRNN(nn.Module):
    def __init__(self,atten_dim,embed_dim,hidden_dim,vocab_size,enc_dim=2048,dropout=0.5,device='cpu'):
        super(DecoderRNN,self).__init__()
        self.atten_dim = atten_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.enc_dim = enc_dim
        self.device = device
        self.ss_prob = 0.0
        self.dropout = nn.Dropout(p=dropout)

        self.atten = SoftAttention(enc_dim=enc_dim,hidden_dim=hidden_dim,atten_dim=atten_dim)
        self.embed = nn.Sequential(
            nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.TD_atten = nn.LSTMCell(input_size=embed_dim+enc_dim+hidden_dim,hidden_size=hidden_dim,bias=True)
        self.language_model = nn.LSTMCell(input_size=enc_dim+hidden_dim,hidden_size=hidden_dim,bias=True)
        self.predict = weight_norm(nn.Linear(in_features=hidden_dim,out_features=vocab_size))
        self.init_weights()

    def init_weights(self):
        self.embed[0].weight.data.uniform_(-0.1,0.1)
        self.predict.weight.data.uniform_(-0.1,0.1)
        self.predict.bias.data.fill_(0)

    def init_hidden_state(self,bsize):
        h = torch.zeros(bsize,self.hidden_dim).to(self.device)
        c = torch.zeros(bsize,self.hidden_dim).to(self.device)
        return h,c

    def forward(self,enc_features,captions,lengths):
        '''
        XE Loss training process.
        :param enc_features:
            when using cnn_extracted features: (bsize,49(7*7pixels),2048)
            when using bottom_up_features: 'fixed':(bsize,36,2048) / 'adaptive':(bsize,bu_len,2048) (currently not support 'adaptive' for BUTD Model)
        :param captions: (bsize,max_len) sorted in length,torch.LongTensor, e.g.:
                [[1(<sta>),24,65,3633,54,234,67,34,12,2(<end>)],    #max_len=10
                [1(<sta>),45,5434,235,12,58,11,2(<end>),0,0],
                ...
                [1(<sta>),24,7534,523,12,2(<end>),0,0,0,0]]
        :param lengths:[9,7,...,5]
                #note that length[i] = len(caption[i])-1 since we skip the 2<end> token when feeding the captions-tensor into the model
                and we skip the 1<sta> token when generating predictions for loss calculation. Thus the total training step in this batch equals to max(lengths)
        :return:packed_predictions: packed_padded_sequence:((total_tokens_nums,vocab_size),indices(could be ignored,not used in training))
        '''
        bsize = enc_features.size(0)
        num_features = enc_features.size(1)   #(bsize,49/36,2048)
        h1,c1 = self.init_hidden_state(bsize)
        h2,c2 = self.init_hidden_state(bsize)
        mean_features = torch.mean(enc_features,dim=1,keepdim=False)    #(bsize,2048)
        predictions = torch.zeros(bsize,max(lengths),self.vocab_size).to(self.device)
        alphas = torch.zeros(bsize,max(lengths),num_features).to(self.device)     #atten weight for each pixel

        for time_step in range(max(lengths)):
            bsize_t = sum([l > time_step for l in lengths])
            if time_step >= 2 and self.ss_prob > 0.0:
                sample_prob = torch.zeros(bsize_t).uniform_(0,1).to(self.device)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = captions[:bsize_t,time_step]
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = captions[:bsize_t,time_step].clone()
                    prob_prev = torch.softmax(preds,dim=1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = captions[:bsize_t,time_step]

            embeddings = self.embed(it) #(bsize_t,embed_dim)
            h1,c1 = self.TD_atten(
                torch.cat([h2[:bsize_t],mean_features[:bsize_t],embeddings],dim=1),
                (h1[:bsize_t],c1[:bsize_t])
            )
            atten_weighted_enc,alpha = self.atten(enc_features[:bsize_t],h1[:bsize_t])
            h2,c2 = self.language_model(
                torch.cat([atten_weighted_enc[:bsize_t],h1[:bsize_t]],dim=1),
                (h2[:bsize_t],c2[:bsize_t])
            )
            preds = self.predict(self.dropout(h2))
            predictions[:bsize_t, time_step, :] = preds
            alphas[:bsize_t, time_step, :] = alpha

        pack_predictions = pack_padded_sequence(predictions, lengths, batch_first=True)
        return pack_predictions, alphas

    def sample(self,enc_features,max_len=20):
        '''
        Use Greedy-search to generate predicted captions
        :param enc_features:
            when using cnn_extracted features: (bsize,49(7*7pixels),2048)
            when using bottom_up_features: 'fixed':(bsize,36,2048) / 'adaptive':(bsize,bu_len,2048) (currently not support 'adaptive' for BUTD Model)
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:sampled_ids:(bsize,max_len)
                alphas:(bsize,max_len,49/36)
        '''
        bsize = enc_features.size(0)
        num_features = enc_features.size(1)   #(bsize,49/36,2048)
        h1,c1 = self.init_hidden_state(bsize)
        h2,c2 = self.init_hidden_state(bsize)
        mean_features = torch.mean(enc_features,dim=1,keepdim=False)    #(bsize,2048)
        captions = torch.LongTensor(bsize,1).fill_(1).to(self.device)
        sampled_ids = []
        alphas = []
        for time_step in range(max_len):
            embeddings = self.embed(captions).squeeze(1)    #(bsize,embed_dim)
            h1,c1 = self.TD_atten(
                torch.cat([h2,mean_features,embeddings],dim=1),
                (h1,c1)
            )
            atten_weighted_enc,alpha = self.atten(enc_features,h1)
            h2,c2 = self.language_model(
                torch.cat([atten_weighted_enc,h1],dim=1),
                (h2,c2)
            )
            preds = self.predict(self.dropout(h2))
            pred_id = preds.max(1)[1]  # (bsize,)
            captions = pred_id.unsqueeze(1)  # (bsize,1)
            sampled_ids.append(captions)
            alphas.append(alpha.unsqueeze(1))  # (bsize,1,196)
        sampled_ids = torch.cat(sampled_ids, dim=1)  # (bsize,max_seq)
        alphas = torch.cat(alphas, dim=1)  # (bsize,max_seq,196)
        return sampled_ids, alphas

    def sample_rl(self,enc_features,max_len=20):
        '''
        Use Monte Carlo method(Random Sampling) to generate sampled predictions for scst training
        :param enc_features:
            when using cnn_extracted features: (bsize,49(7*7pixels),2048)
            when using bottom_up_features: 'fixed':(bsize,36,2048) / 'adaptive':(bsize,bu_len,2048) (currently not support 'adaptive' for BUTD Model)
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:seq:(bsize,max_len)
                seqLogprobs:(bsize,max_len)
        '''
        bsize = enc_features.size(0)
        num_features = enc_features.size(1)   #(bsize,49,2048)
        h1,c1 = self.init_hidden_state(bsize)
        h2,c2 = self.init_hidden_state(bsize)
        mean_features = torch.mean(enc_features,dim=1,keepdim=False)    #(bsize,2048)
        its = torch.LongTensor(bsize,1).fill_(1).to(self.device)
        seq = torch.zeros(bsize,max_len,dtype=torch.long).to(self.device)
        seqLogprobs = torch.zeros(bsize,max_len).to(self.device)
        for time_step in range(max_len):
            embeddings = self.embed(its).squeeze(1)    #(bsize,embed_dim)
            h1,c1 = self.TD_atten(
                torch.cat([h2,mean_features,embeddings],dim=1),
                (h1,c1)
            )
            atten_weighted_enc,alpha = self.atten(enc_features,h1)
            h2,c2 = self.language_model(
                torch.cat([atten_weighted_enc,h1],dim=1),
                (h2,c2)
            )
            preds = self.predict(self.dropout(h2))
            logprobs = Fun.log_softmax(preds, dim=1)
            prob_prev = torch.exp(logprobs)
            its = torch.multinomial(prob_prev, num_samples=1)  # sample a word  #(bsize,1)
            sampleLogprobs = logprobs.gather(1, its)  # gather the logprobs at sampled positions   #(bsize,1)
            its = its.clone()
            if time_step == 0:
                unfinished = abs(its - 2) > 0
            else:
                unfinished = unfinished * (abs(its - 2) > 0)
            its = its * unfinished.type_as(its)
            seq[:, time_step] = its.view(-1)
            seqLogprobs[:, time_step] = sampleLogprobs.view(-1)
            if unfinished.sum() == 0: break
        return seq,seqLogprobs

    def beam_search_sample(self,enc_features,beam_size=5):
        '''
        Use Beam search method to generate predicted captions, asserting bsize=1
        :param enc_features:
                when using cnn_extracted features: (bsize,49(7*7pixels),2048)
                when using bottom_up_features: 'fixed':(bsize,36,2048) / 'adaptive':(bsize,bu_len,2048) (currently not support 'adaptive' for BUTD Model)
        :param beam_size: beam numbers scalar
        :return:seq_tensor:(1,sentence_len)
                alphas_tensor:(1,sentence_len,49/36)
        '''
        num_features = enc_features.size(1)
        k = beam_size
        k_prev_words = torch.LongTensor(k,1).fill_(1).to(self.device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k,1).to(self.device)
        mean_features = torch.mean(enc_features,dim=1,keepdim=False)    #(bsize,2048)
        enc_features = enc_features.expand(k,enc_features.shape[1],enc_features.shape[2])   #(k,h*w,2048)
        mean_features = mean_features.expand(k,mean_features.shape[1])    #(k,2048)
        complete_seqs = list()
        complete_seqs_scores = list()
        alphas = torch.zeros(k, 1, num_features).to(self.device)
        complete_alphas = list()

        step = 1
        max_step_limit = 50
        h1,c1 = self.init_hidden_state(bsize=k)    #(k,hidden_dim)
        h2,c2 = self.init_hidden_state(bsize=k)
        while step<=max_step_limit:
            embeddings = self.embed(k_prev_words).squeeze(1)    #(s=active_beam_num,embed_dim)
            h1,c1 = self.TD_atten(torch.cat([h2,mean_features,embeddings],dim=1),(h1,c1))
            atten_weighted_enc, alpha = self.atten(enc_features,h1)  #(s,2048),(s,196)

            h2,c2 = self.language_model(torch.cat((atten_weighted_enc,h1),dim=1),(h2,c2))     #(s,hidden_dim)

            scores = self.predict(self.dropout(h2))
            scores = Fun.log_softmax(scores,dim=1)
            scores = top_k_scores.expand_as(scores) + scores    #(s,vocab_size)
            if step == 1:
                top_k_scores,top_k_words = scores[0].topk(k,0,True,True) #(s)
            else:
                top_k_scores,top_k_words = scores.view(-1).topk(k,0,True,True)
            prev_word_inds = top_k_words / self.vocab_size
            next_word_inds = top_k_words % self.vocab_size
            seqs = torch.cat([seqs[prev_word_inds],next_word_inds.unsqueeze(1)],dim=1)  #(s,step+1)
            alphas = torch.cat([alphas[prev_word_inds],alpha.unsqueeze(1)],dim=1)   #(s,step+1,196)

            incomplete_inds = [ind for ind,next_word in enumerate(next_word_inds) if next_word != 2]
            complete_inds = list(set(range(len(next_word_inds)))-set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                complete_alphas.extend(alphas[complete_inds].tolist())

            k -= len(complete_inds)
            if k==0:
                break

            seqs = seqs[incomplete_inds]
            enc_features = enc_features[prev_word_inds[incomplete_inds]]
            mean_features = mean_features[prev_word_inds[incomplete_inds]]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            step += 1

        if len(complete_seqs)>0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
            seq_tensor = torch.Tensor(seq).unsqueeze(0)
            complete_alpha = complete_alphas[i]
            alpha_tensor = torch.Tensor(complete_alpha).to(self.device).unsqueeze(0)    #(1,len,196)
        else:
            i = torch.max(top_k_scores,dim=0)[1].item()
            seq_tensor = seqs[i].unsqueeze(0)
            alpha_tensor = alphas[i].unsqueeze(0)
        alpha_tensor = alpha_tensor[:,1:,:]         #skip the first time_step where alpha=torch.zeros since it's just for convience of writing the beamsearch code

        return seq_tensor,alpha_tensor

#---------------------------Model Class---------------------------------------#
class BUTDSpatial_Captioner(nn.Module):
    '''
    'Spatial' model using 7*7*2048 cnn extracted visual features, thus having EncoderCNN
    '''
    def __init__(self,encoded_img_size,atten_dim,embed_dim,hidden_dim,vocab_size,dropout=0.5,device='cpu'):
        super(BUTDSpatial_Captioner,self).__init__()
        self.encoder = EncoderCNN(encoded_img_size=encoded_img_size)
        self.decoder = DecoderRNN(atten_dim=atten_dim,embed_dim=embed_dim,hidden_dim=hidden_dim,vocab_size=vocab_size,dropout=dropout,device=device)
        self.cnn_finetune(flag=False)

    def get_param_groups(self,lr_dict):
        cnn_extractor_params = list(filter(lambda p: p.requires_grad, self.encoder.feature_extractor.parameters()))
        captioner_params = list(self.decoder.parameters())
        assert lr_dict.__contains__('cnn_ft_lr')
        param_groups = [
            {'params':captioner_params,'lr':lr_dict['lr']},
            {'params':cnn_extractor_params,'lr':lr_dict['cnn_ft_lr']}
        ]
        return param_groups

    def cnn_finetune(self, flag=False):
        '''
        we only fine tune on the last layer of resnet-101
        :param flag: enable cnn fine_tune when set true.
        '''
        if flag:
            for module in list(self.encoder.feature_extractor.children())[7:]:
                for param in module.parameters():
                    param.requires_grad = True
        else:
            for params in self.encoder.feature_extractor.parameters():
                params.requires_grad = False

    def forward(self,visual_inputs,captions,lengths):
        '''
        XE Loss training process.
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize,3,224,224)}
            dict of available visual features. when using BUTDSpatial model, only raw img tensors are required
        :param captions: (bsize,max_len) sorted in length,torch.LongTensor, e.g.:
                [[1(<sta>),24,65,3633,54,234,67,34,12,2(<end>)],    #max_len=10
                [1(<sta>),45,5434,235,12,58,11,2(<end>),0,0],
                ...
                [1(<sta>),24,7534,523,12,2(<end>),0,0,0,0]]
        :param lengths:[9,7,...,5]
                #note that length[i] = len(caption[i])-1 since we skip the 2<end> token when feeding the captions-tensor into the model
                and we skip the 1<sta> token when generating predictions for loss calculation. Thus the total training step in this batch equals to max(lengths)
        :return:packed_predictions: packed_padded_sequence:((total_tokens_nums,vocab_size),indices(could be ignored,not used in training))
        '''
        images = visual_inputs['img_tensors']   #(bsize,3,224,224)
        enc_features = self.encoder(images)     #(bsize,49,2048)
        packed_predictions,alphas = self.decoder(enc_features,captions,lengths)
        return packed_predictions

    def sampler(self,visual_inputs,max_len=20):
        '''
        Use Greedy-search to generate predicted captions
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize,3,224,224)}
            dict of available visual features. when using BUTDSpatial model, only raw img tensors are required
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:sampled_ids:(bsize,max_len)
        '''
        images = visual_inputs['img_tensors']   #(bsize,3,224,224)
        enc_features = self.encoder(images)
        sampled_ids,alphas = self.decoder.sample(enc_features,max_len)
        return sampled_ids

    def sampler_rl(self,visual_inputs,max_len=20):
        '''
        Use Monte Carlo method(Random Sampling) to generate sampled predictions for scst training
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize,3,224,224)}
            dict of available visual features. when using BUTDSpatial model, only raw img tensors are required
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:seq:(bsize,max_len)
                seqLogprobs:(bsize,max_len)
        '''
        images = visual_inputs['img_tensors']   #(bsize,3,224,224)
        enc_features = self.encoder(images)
        seq,seqLogprobs = self.decoder.sample_rl(enc_features=enc_features,max_len=max_len)
        return seq,seqLogprobs

    def beam_search_sampler(self,visual_inputs,beam_size=5):
        '''
        Use Beam search method to generate predicted captions, asserting bsize=1
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize,3,224,224)}
            dict of available visual features. when using BUTDSpatial model, only raw img tensors are required
        :param beam_size: beam numbers scalar
        :return: sampled_ids:(bsize=1,sentence_len)
        '''
        images = visual_inputs['img_tensors']   #(bsize,3,224,224)
        enc_features = self.encoder(images)
        sampled_ids,alphas = self.decoder.beam_search_sample(enc_features=enc_features,beam_size=beam_size)
        return sampled_ids

    def eval_test_image(self,visual_inputs,caption_vocab,max_len=20,eval_beam_size=-1):
        '''
        Tests on single given image.
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize=1,3,224,224)}
            dict of available visual features. when using BUTDSpatial model, only raw img tensors are required
        :param caption_vocab: pkl-file. used to translate the generated sentence into human language.
        :param max_len: maximum length(or time_step) of the generated sentence
        :param eval_beam_size: beam numbers scalar
        :return:caption: generated caption for given image.
                additional output: e.g. attention weights over different visual features at each time step during training.(not used in NIC)
        '''
        image = visual_inputs['img_tensors']   #(bsize=1,3,224,224)
        assert image.size(0) == 1
        enc_features = self.encoder(image)
        if eval_beam_size != -1:
            sampled_ids,alphas = self.decoder.beam_search_sample(enc_features=enc_features,beam_size=eval_beam_size)
        else:
            sampled_ids,alphas = self.decoder.sample(enc_features=enc_features,max_len=max_len)
        caption_ids = sampled_ids[0].cpu().detach().numpy()
        caption = []
        for word_id in caption_ids:
            word = caption_vocab.ix2word[word_id]
            if word == '<end>':
                break
            elif word != '<sta>':
                caption.append(word)
        return caption,[alphas]

#---------------------------------------------------------------------------------------------#
class BUTDDetection_Captioner(nn.Module):
    '''
    'Detection' model using pretrained faster-rcnn bottom-up features
    '''
    def __init__(self,atten_dim,embed_dim,hidden_dim,vocab_size,dropout=0.5,device='cpu'):
        super(BUTDDetection_Captioner,self).__init__()
        self.decoder = DecoderRNN(atten_dim=atten_dim,embed_dim=embed_dim,hidden_dim=hidden_dim,vocab_size=vocab_size,dropout=dropout,device=device)

    def get_param_groups(self,lr_dict):
        captioner_params = list(self.decoder.parameters())
        param_groups = [
            {'params':captioner_params,'lr':lr_dict['lr']}
        ]
        return param_groups

    def forward(self,visual_inputs,captions,lengths):
        '''
        XE Loss training process.
        :param visual_inputs: {'bu_feats':torch.FloatTensor(bsize,36,2048),'bu_bboxes':[np.ndarry(36,4)],'bu_mask':None}
            dict of available visual features. when using BUTDDetection model, 'fixed' bottom_up features are required
            Since BUTD Models did not use the 'adaptive' bu_feats, bu_mask is not used.
        :param captions: (bsize,max_len) sorted in length,torch.LongTensor, e.g.:
                [[1(<sta>),24,65,3633,54,234,67,34,12,2(<end>)],    #max_len=10
                [1(<sta>),45,5434,235,12,58,11,2(<end>),0,0],
                ...
                [1(<sta>),24,7534,523,12,2(<end>),0,0,0,0]]
        :param lengths:[9,7,...,5]
                #note that length[i] = len(caption[i])-1 since we skip the 2<end> token when feeding the captions-tensor into the model
                and we skip the 1<sta> token when generating predictions for loss calculation. Thus the total training step in this batch equals to max(lengths)
        :return:packed_predictions: packed_padded_sequence:((total_tokens_nums,vocab_size),indices(could be ignored,not used in training))
        '''
        bottom_up_features = visual_inputs['bu_feats']
        packed_predictions,alphas = self.decoder(bottom_up_features,captions,lengths)
        return packed_predictions

    def sampler(self,visual_inputs,max_len=20):
        '''
        Use Greedy-search to generate predicted captions
        :param visual_inputs: {'bu_feats':torch.FloatTensor(bsize,36,2048),'bu_bboxes':[np.ndarry(36,4)],'bu_mask':None}
            dict of available visual features. when using BUTDDetection model, 'fixed' bottom_up features are required
            Since BUTD Models did not use the 'adaptive' bu_feats, bu_mask is not used.
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:sampled_ids:(bsize,max_len)
        '''
        bottom_up_features = visual_inputs['bu_feats']
        sampled_ids,alphas = self.decoder.sample(bottom_up_features,max_len)
        return sampled_ids

    def sampler_rl(self,visual_inputs,max_len=20):
        '''
        Use Monte Carlo method(Random Sampling) to generate sampled predictions for scst training
        :param visual_inputs: {'bu_feats':torch.FloatTensor(bsize,36,2048),'bu_bboxes':[np.ndarry(36,4)],'bu_mask':None}
            dict of available visual features. when using BUTDDetection model, 'fixed' bottom_up features are required
            Since BUTD Models did not use the 'adaptive' bu_feats, bu_mask is not used.
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:seq:(bsize,max_len)
                seqLogprobs:(bsize,max_len)
        '''
        bottom_up_features = visual_inputs['bu_feats']
        seq,seqLogprobs = self.decoder.sample_rl(enc_features=bottom_up_features,max_len=max_len)
        return seq,seqLogprobs

    def beam_search_sampler(self,visual_inputs,beam_size=5):
        '''
        Use Beam search method to generate predicted captions, asserting bsize=1
        :param visual_inputs: {'bu_feats':torch.FloatTensor(bsize,36,2048),'bu_bboxes':[np.ndarry(36,4)],'bu_mask':None}
            dict of available visual features. when using BUTDDetection model, 'fixed' bottom_up features are required
            Since BUTD Models did not use the 'adaptive' bu_feats, bu_mask is not used.
        :param beam_size: beam numbers scalar
        :return: sampled_ids:(bsize=1,sentence_len)
        '''
        bottom_up_features = visual_inputs['bu_feats']
        sampled_ids,alphas = self.decoder.beam_search_sample(enc_features=bottom_up_features,beam_size=beam_size)
        return sampled_ids

    def eval_test_image(self,visual_inputs,caption_vocab,max_len=20,eval_beam_size=-1):
        '''
        Tests on single given image.
        :param visual_inputs: {'bu_feats':torch.FloatTensor(bsize,36,2048),'bu_bboxes':[np.ndarry(36,4)], 'bu_mask':None}
            dict of available visual features. when using BUTDDetection model, 'fixed' bottom_up features are required
            Since BUTD Models did not use the 'adaptive' bu_feats, bu_mask is not used.
        :param caption_vocab: pkl-file. used to translate the generated sentence into human language.
        :param max_len: maximum length(or time_step) of the generated sentence
        :param eval_beam_size: beam numbers scalar
        :return:caption: generated caption for given image.
                additional output: e.g. attention weights over different visual features at each time step during training.(not used in NIC)
        '''
        bottom_up_features = visual_inputs['bu_feats']
        assert bottom_up_features.size(0) == 1
        if eval_beam_size != -1:
            sampled_ids,alphas = self.decoder.beam_search_sample(enc_features=bottom_up_features,beam_size=eval_beam_size)
        else:
            sampled_ids,alphas = self.decoder.sample(enc_features=bottom_up_features,max_len=max_len)
        caption_ids = sampled_ids[0].cpu().detach().numpy()
        caption = []
        for word_id in caption_ids:
            word = caption_vocab.ix2word[word_id]
            if word == '<end>':
                break
            elif word != '<sta>':
                caption.append(word)
        return caption,[alphas]