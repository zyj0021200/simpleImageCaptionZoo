import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,PackedSequence
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as Fun
import copy
import math

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gain = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)         #(bsize,feature_nums,feature_dims)
        std = x.std(-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.eps) + self.bias

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout_sc=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(feature_dim=size)
        self.dropout = nn.Dropout(p=dropout_sc)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x))[0])

def DotProductAttention(query,key,value,mask=None,dropout=None):
    '''
    Compute 'Scaled Dot Product Attention'
    when using cnn_extracted features: kv_feature_nums=49
    when using bottom_up_features: 'fixed': kv_feature_nums=36 / 'adaptive': kv_feature_nums=bu_len
        bu_len(10~100) is the maximum number of bu_features each image has in this batch
    when doing image feature refinement in AoA_refine_core: q_feature_nums=kv_feature_nums
    when doing sentence decoding in AoA_decoder: q_feature_nums=1(ht on time_step_i)
    :param query:(bsize,num_heads,q_feature_nums,d_q)
    :param key:(bsize,num_heads,kv_feature_nums,d_k)
    :param value:(bsize,num_heads,kv_feature_nums,d_v)
    :param mask:
        when using cnn_extracted features: None
        when using bottom_up_features:
            'fixed': None
            'adaptive': (bsize,1(will be expanded to num_heads),1(will be expanded to query_feature_nums),kv_feature_nums=bu_len)
    :param dropout:torch.nn.dropout()
    :return:value_:(bsize,num_heads,q_feature_nums,d_v)
            p_atten:(bsize,num_heads,q_feature_nums,kv_feature_nums)
    '''
    d_k = key.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)      #(bsize,num_heads,q_feature_nums,kv_feature_nums)
    if mask is not None:
        scores = scores.masked_fill(mask==0,-1e9)
    p_atten = torch.softmax(scores,dim=-1)  #(bsize,num_heads,q_feature_nums,kv_feature_nums)
    if dropout is not None:
        p_atten = dropout(p_atten)
    value_ = torch.matmul(p_atten,value)    #(bsize,num_heads,q_feature_nums,dv)
    return value_,p_atten

class AoABlock(nn.Module):
    def __init__(self,num_heads,d_model,dropout_aoa=0.3,dropout_dotAtten=0.1):
        super(AoABlock,self).__init__()
        #make sure the input features can be divided into N heads
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_q = d_model // num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.linear_Q = nn.Linear(in_features=d_model,out_features=d_model)
        self.linear_K = nn.Linear(in_features=d_model,out_features=d_model)
        self.linear_V = nn.Linear(in_features=d_model,out_features=d_model)
        self.aoa_module = nn.Sequential(nn.Linear(in_features=2*d_model,out_features=2*d_model),nn.GLU())
        if dropout_aoa>0:
            self.dropout_aoa = nn.Dropout(p=dropout_aoa)
        else:
            self.dropout_aoa = lambda x:x
        self.atten_dropout = nn.Dropout(p=dropout_dotAtten)

    def forward(self,query,key,value,mask=None):
        '''
        Compute AoA^E(f_mh-att,Q,K,V)
        when using cnn_extracted features: kv_feature_nums=49
        when using bottom_up_features: 'fixed': kv_feature_nums=36 / 'adaptive': kv_feature_nums=bu_len
            bu_len(10~100) is the maximum number of bu_features each image has in this batch
        when doing image feature refinement in AoA_refine_core: q_feature_nums=kv_feature_nums
        when doing sentence decoding in AoA_decoder: q_feature_nums=1(ht on time_step_i)
        :param query:(bsize,q_feature_nums,d_model)
        :param key:(bsize,kv_feature_nums,d_model)
        :param value:(bsize,kv_feature_nums,d_model)
        :param mask:
                when using cnn_extracted features: None
                when using bottom_up_features: 'fixed': None / 'adaptive': (bsize,kv_feature_nums=bu_len)
        :return:x:(bsize,q_feature_nums,d_model)
                atten:(bsize,q_feature_nums,kv_feature_nums)
        '''
        bsize = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)    #(bsize,kv_feature_nums)->(bsize,1,kv_feature_nums)
            mask = mask.unsqueeze(1)    #(bsize,1,kv_feature_nums)->(bsize,1,1,kv_feature_nums)
        # (bsize,feature_nums,d_model)->(bsize,feature_nums,d_model)->
        # ->(bsize,feature_nums,num_heads,d_q/k/v)->(bsize,num_heads,feature_nums,d_q/k/v) d_k=d_q=d_v=d_model//num_heads
        query_p = self.linear_Q(query).view(bsize,-1,self.num_heads,self.d_q).transpose(1,2)
        key_p = self.linear_K(key).view(bsize, -1, self.num_heads, self.d_k).transpose(1, 2)
        value_p = self.linear_V(value).view(bsize, -1, self.num_heads, self.d_v).transpose(1, 2)
        x,atten = DotProductAttention(query=query_p,key=key_p,value=value_p,mask=mask,dropout=self.atten_dropout) #(bsize,num_heads,q_feature_nums,d_v)/(bsize,num_heads,q_feature_nums,kv_feature_nums)
        x = x.transpose(1,2).contiguous().view(bsize,-1,self.num_heads*self.d_v)    #(bsize,num_heads,q_feature_nums,dv)->(bsize,q_feature_nums,num_heads,dv)->(bsize,query_feature_nums,num_heads*dv=d_model)
        x = self.aoa_module(self.dropout_aoa(torch.cat([x,query],dim=-1)))     #(bsize,q_feature_nums,d_model)
        mean_atten = atten.mean(dim=1,keepdim=False) #(bsize,num_heads,q_feature_nums,kv_feature_nums)->(bsize,q_feature_nums,kv_feature_nums)
        return x,mean_atten

class AoA_Refine_Block(nn.Module):
    '''
    AoABlock + Sub_Connection = full AoA_Refine_Block, will be stacked 6 times in AoA_Refine_Core
    '''
    def __init__(self, num_heads, d_model=1024, dropout_aoa=0.3,dropout_sc=0.1,dropout_dotAtten=0.1):
        super(AoA_Refine_Block,self).__init__()
        self.aoa_block = AoABlock(
            num_heads=num_heads,
            d_model=d_model,
            dropout_aoa=dropout_aoa,
            dropout_dotAtten=dropout_dotAtten
        )
        self.sublayer = SublayerConnection(size=d_model, dropout_sc=dropout_sc)

    def forward(self,x,bu_mask):
        out = self.sublayer(x=x,sublayer=lambda x: self.aoa_block(query=x,key=x,value=x,mask=bu_mask))
        return out

class AoA_Refine_Core(nn.Module):
    def __init__(self, num_heads, d_model=1024, dropout_aoa=0.3,dropout_sc=0.1,dropout_dotAtten=0.1):
        super(AoA_Refine_Core,self).__init__()
        aoa_sub_layer = AoA_Refine_Block(
            num_heads=num_heads,
            d_model=d_model,
            dropout_aoa=dropout_aoa,
            dropout_sc=dropout_sc,
            dropout_dotAtten=dropout_dotAtten
        )
        self.aoa_layers = clones(aoa_sub_layer,N=6)
        self.norm = LayerNorm(feature_dim=d_model)

    def forward(self,x,bu_mask=None):
        '''
        Encoder with AoA
        :param x: (bsize,q_feature_nums,d_model)
        :param bu_mask: (bsize,kv_feature_nums)
        :return:
        '''
        for layer in self.aoa_layers:
            x = layer(x=x,bu_mask=bu_mask)
        return self.norm(x)

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
        self.GAP = nn.AdaptiveAvgPool2d((1,1))

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

class AoA_Decoder(nn.Module):
    def __init__(self,hidden_dim,num_heads,embed_dim,vocab_size,d_model,dropout=0.5,device='cpu'):
        super(AoA_Decoder,self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.ss_prob = 0.0
        self.lstm = nn.LSTMCell(input_size=self.embed_dim+hidden_dim,hidden_size=hidden_dim)      #(x_t=[WeIIt,a_avg+c_t-1])
        self.aoa_block = AoABlock(num_heads=num_heads,d_model=d_model,dropout_aoa=0)
        self.embed = nn.Sequential(
            nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.h_norm = LayerNorm(feature_dim=hidden_dim)
        self.predict = weight_norm(nn.Linear(in_features=hidden_dim,out_features=vocab_size))
        self.ctx_dropout = nn.Dropout(p=dropout)
        self.out_dropout = nn.Dropout(p=dropout)
        self.device = device
        self.init_weights()

    def init_weights(self):
        self.embed[0].weight.data.uniform_(-0.1,0.1)
        self.predict.weight.data.uniform_(-0.1,0.1)
        self.predict.bias.data.fill_(0)

    def init_hidden_state(self,bsize):
        h = torch.zeros(bsize,self.hidden_dim).to(self.device)
        m = torch.zeros(bsize,self.hidden_dim).to(self.device)
        ctx = torch.zeros(bsize,self.hidden_dim).to(self.device)
        return h,m,ctx

    def forward(self,enc_features,captions,lengths,bu_masks=None):
        '''
        :param enc_features:
                when using cnn_extracted features: (bsize,49(7*7pixels),2048)
                when using bottom_up_features: 'fixed':(bsize,36,2048) / 'adaptive':(bsize,bu_len,2048)
                    bu_len(10~100) is the maximum number of bu_features each image has in this batch
        :param bu_mask:
                when using cnn_extracted features: None
                when using bottom_up_features: 'fixed':None / 'adaptive':(bsize,bu_len)
        :param captions: (bsize,max_len) sorted in length,torch.LongTensor
                [[1(<sta>),24,65,3633,54,234,67,34,12,2(<end>)],    #max_len=10
                [1(<sta>),45,5434,235,12,58,11,2(<end>),0,0],
                ...
                [1(<sta>),24,7534,523,12,2(<end>),0,0,0,0]]
        :param lengths:[9,7,...,5]
                #note that length[i] = len(caption[i])-1 since we skip the 2<end> token when feeding the captions-tensor into the model
                and we skip the 1<sta> token when generating predictions for loss calculation. Thus the total training step in this batch equals to max(lengths)
        :return:packed_padded_predictions: ((total_tokens_nums,vocab_size),indices(ignore))
        '''
        bsize = enc_features.size(0)
        num_features = enc_features.size(1)   #(bsize,49/36/10~100,1024=hidden_dim=d_model)
        if bu_masks is None:
            mean_features = torch.mean(enc_features, dim=1) #(bsize,1024)
        else:
            mean_features = (torch.sum(enc_features * bu_masks.unsqueeze(-1), 1) / torch.sum(bu_masks.unsqueeze(-1), 1))
        h,m,ctx = self.init_hidden_state(bsize)
        #ctx vector is initialized to zeros at the begining step (bsize,hidden_dim) so it is equal to h&m
        predictions = torch.zeros(bsize,max(lengths),self.vocab_size).to(self.device)

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
            h,m = self.lstm(
                torch.cat([embeddings,mean_features[:bsize_t]+self.ctx_dropout(ctx[:bsize_t])],dim=1),
                (h[:bsize_t],m[:bsize_t])
            )
            if bu_masks is not None:
                bu_masks_this_step = bu_masks[:bsize_t]
            else:
                bu_masks_this_step = None
            ctx,_ = self.aoa_block(
                query=self.h_norm(h.unsqueeze(1)),
                key=enc_features[:bsize_t],
                value=enc_features[:bsize_t],
                mask=bu_masks_this_step
            ) #(bsize,1,d_model)
            ctx = ctx.squeeze(1)    #(bsize,d_model=hidden_dim)
            preds = self.predict(self.out_dropout(ctx))
            predictions[:bsize_t, time_step, :] = preds

        pack_predictions = pack_padded_sequence(predictions, lengths, batch_first=True)
        return pack_predictions

    def sample(self,enc_features,max_len=20,bu_masks=None):
        '''
        Use Greedy-search to generate predicted captions
        :param enc_features:
                when using cnn_extracted features: (bsize,49(7*7pixels),2048)
                when using bottom_up_features: 'fixed':(bsize,36,2048) / 'adaptive':(bsize,bu_len,2048)
                    bu_len(10~100) is the maximum number of bu_features each image has in this batch
        :param bu_mask:
                when using cnn_extracted features: None
                when using bottom_up_features: 'fixed':None / 'adaptive':(bsize,bu_len)
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:sampled_ids:(bsize,max_len)
                alphas:(bsize,max_len,49/36/bu_len)
        '''
        bsize = enc_features.size(0)
        num_features = enc_features.size(1)   #(bsize,49,2048)
        if bu_masks is None:
            mean_features = torch.mean(enc_features, dim=1) #(bsize,1024)
        else:
            mean_features = (torch.sum(enc_features * bu_masks.unsqueeze(-1), 1) / torch.sum(bu_masks.unsqueeze(-1), 1))
        h,m,ctx = self.init_hidden_state(bsize)
        captions = torch.LongTensor(bsize,1).fill_(1).to(self.device)
        sampled_ids = []
        alphas = []
        for time_step in range(max_len):
            embeddings = self.embed(captions).squeeze(1)    #(bsize,embed_dim)
            h,m = self.lstm(
                torch.cat([embeddings,mean_features+self.ctx_dropout(ctx)],dim=1),
                (h,m)
            )
            if bu_masks is not None:
                bu_masks_this_step = bu_masks
            else:
                bu_masks_this_step = None
            ctx,alpha = self.aoa_block(
                query=self.h_norm(h.unsqueeze(1)),
                key=enc_features,
                value=enc_features,
                mask=bu_masks_this_step
            ) #(bsize,1,d_model)/(bsize,1,36)
            ctx = ctx.squeeze(1)    #(bsize,d_model=hidden_dim)
            preds = self.predict(self.out_dropout(ctx))
            pred_id = preds.max(1)[1]  # (bsize,)
            captions = pred_id.unsqueeze(1)  # (bsize,1)
            sampled_ids.append(captions)
            alphas.append(alpha)  # (bsize,1,36)

        sampled_ids = torch.cat(sampled_ids, dim=1)  # (bsize,max_seq)
        alphas = torch.cat(alphas, dim=1)  # (bsize,max_seq,196)
        return sampled_ids, alphas

    def sample_rl(self,enc_features,max_len=20,bu_masks=None):
        '''
        Use Monte Carlo method(Random Sampling) to generate sampled predictions for scst training
        :param enc_features:
                when using cnn_extracted features: (bsize,49(7*7pixels),2048)
                when using bottom_up_features: 'fixed':(bsize,36,2048) / 'adaptive':(bsize,bu_len,2048)
                    bu_len(10~100) is the maximum number of bu_features each image has in this batch
        :param bu_mask:
                when using cnn_extracted features: None
                when using bottom_up_features: 'fixed':None / 'adaptive':(bsize,bu_len)
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:seq:(bsize,max_len)
                seqLogprobs:(bsize,max_len)
        '''
        bsize = enc_features.size(0)
        num_features = enc_features.size(1)   #(bsize,49/36,1024)
        if bu_masks is None:
            mean_features = torch.mean(enc_features, dim=1) #(bsize,1024)
        else:
            mean_features = (torch.sum(enc_features * bu_masks.unsqueeze(-1), 1) / torch.sum(bu_masks.unsqueeze(-1), 1))
        h,m,ctx = self.init_hidden_state(bsize)
        its = torch.LongTensor(bsize,1).fill_(1).to(self.device)
        seq = torch.zeros(bsize,max_len,dtype=torch.long).to(self.device)
        seqLogprobs = torch.zeros(bsize,max_len).to(self.device)
        for time_step in range(max_len):
            embeddings = self.embed(its).squeeze(1)    #(bsize,embed_dim)
            h,m = self.lstm(
                torch.cat([embeddings,mean_features+self.ctx_dropout(ctx)],dim=1),
                (h,m)
            )
            if bu_masks is not None:
                bu_masks_this_step = bu_masks
            else:
                bu_masks_this_step = None
            ctx,alpha = self.aoa_block(
                query=self.h_norm(h.unsqueeze(1)),
                key=enc_features,
                value=enc_features,
                mask=bu_masks_this_step
            ) #(bsize,1,d_model)
            ctx = ctx.squeeze(1)    #(bsize,d_model=hidden_dim)
            preds = self.predict(self.out_dropout(ctx))
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

    def beam_search_sample(self,enc_features,beam_size=5,bu_masks=None):
        '''
        Use Beam search method to generate predicted captions, asserting bsize=1
        :param enc_features:
                when using cnn_extracted features: (bsize,49(7*7pixels),2048)
                when using bottom_up_features: 'fixed':(bsize,36,2048) / 'adaptive':(bsize,bu_len,2048)
                    bu_len(10~100) is the maximum number of bu_features each image has in this batch
        :param bu_mask:
                when using cnn_extracted features: None
                when using bottom_up_features: 'fixed':None / 'adaptive':(bsize,bu_len)
        :param beam_size: beam numbers scalar
        :return:seq_tensor:(1,sentence_len)
                alpha_tensor:(1,sentence_len,49/36/bu_len)
        '''
        num_features = enc_features.size(1)
        k = beam_size
        k_prev_words = torch.LongTensor(k,1).fill_(1).to(self.device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k,1).to(self.device)
        if bu_masks is None:
            mean_features = torch.mean(enc_features, dim=1) #(bsize,1024)
        else:
            mean_features = (torch.sum(enc_features * bu_masks.unsqueeze(-1), 1) / torch.sum(bu_masks.unsqueeze(-1), 1))

        enc_features = enc_features.expand(k,enc_features.shape[1],enc_features.shape[2])   #(k,h*w/36,1024)
        mean_features = mean_features.expand(k,mean_features.shape[1])    #(k,1024)
        complete_seqs = list()
        complete_seqs_scores = list()
        alphas = torch.zeros(k, 1, num_features).to(self.device)
        complete_alphas = list()

        step = 1
        max_step_limit = 50
        h,m,ctx = self.init_hidden_state(bsize=beam_size)
        while step<=max_step_limit:
            embeddings = self.embed(k_prev_words).squeeze(1)    #(s=active_beam_num,embed_dim)

            h,m = self.lstm(
                torch.cat([embeddings,mean_features+self.ctx_dropout(ctx)],dim=1),
                (h,m)
            )
            if bu_masks is not None:
                bu_masks_this_step = bu_masks
            else:
                bu_masks_this_step = None
            ctx,alpha = self.aoa_block(
                query=self.h_norm(h.unsqueeze(1)),
                key=enc_features,
                value=enc_features,
                mask=bu_masks_this_step
            )   #(s,1,d_model=hidden_dim=1024)/(s,1,36)
            ctx = ctx.squeeze(1)    #(s,d_model)
            scores = self.predict(self.out_dropout(ctx))
            scores = Fun.log_softmax(scores,dim=1)
            scores = top_k_scores.expand_as(scores) + scores    #(s,vocab_size)
            if step == 1:
                top_k_scores,top_k_words = scores[0].topk(k,0,True,True) #(s)
            else:
                top_k_scores,top_k_words = scores.view(-1).topk(k,0,True,True)
            prev_word_inds = top_k_words / self.vocab_size
            next_word_inds = top_k_words % self.vocab_size
            seqs = torch.cat([seqs[prev_word_inds],next_word_inds.unsqueeze(1)],dim=1)  #(s,step+1)
            alphas = torch.cat([alphas[prev_word_inds],alpha],dim=1)   #(s,step+1,196)

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
            ctx = ctx[prev_word_inds[incomplete_inds]]
            h = h[prev_word_inds[incomplete_inds]]
            m = m[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            step += 1

        if len(complete_seqs)>0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
            seq_tensor = torch.Tensor(seq).unsqueeze(0)
            complete_alpha = complete_alphas[i]
            alpha_tensor = torch.Tensor(complete_alpha).to(self.device).unsqueeze(0)    #(1,len,36)
        else:
            i = torch.max(top_k_scores,dim=0)[1].item()
            seq_tensor = seqs[i].unsqueeze(0)
            alpha_tensor = alphas[i].unsqueeze(0)
        alpha_tensor = alpha_tensor[:,1:,:]         #skip the first time_step where alpha=torch.zeros since it's just for convience of writing the beamsearch code

        return seq_tensor,alpha_tensor

#--------------------------------Model Class----------------------------------#
class AoASpatial_Captioner(nn.Module):
    def __init__(self,encoded_img_size,vocab_size,num_heads=8,hidden_dim=1024,embed_dim=1024,dropout_aoa=0.3,dropout_prob=0.5,device='cpu'):
        super(AoASpatial_Captioner,self).__init__()
        #img_embed_dim=hidden_dim=1024
        self.encoder = EncoderCNN(encoded_img_size=encoded_img_size)
        self.img_feats_porjection = nn.Sequential(
            nn.Linear(in_features=2048,out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.aoa_refine = AoA_Refine_Core(num_heads=num_heads,d_model=hidden_dim,dropout_aoa=dropout_aoa)
        self.decoder = AoA_Decoder(hidden_dim=hidden_dim,num_heads=num_heads,embed_dim=embed_dim,vocab_size=vocab_size,d_model=hidden_dim,dropout=dropout_prob,device=device)
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
            dict of available visual features. when using AoASpatial model, only raw img tensors are required
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
        images = visual_inputs['img_tensors']    #(bsize,3,224,224)
        enc_features = self.encoder(images)     #(bsize,49,2048)
        projected_enc_features = self.img_feats_porjection(enc_features)    #(bsize,49,1024)
        refined_features = self.aoa_refine(x=projected_enc_features)        #(bsize,49,1024)
        packed_predictions = self.decoder(refined_features,captions,lengths)
        return packed_predictions

    def sampler(self,visual_inputs,max_len=20):
        '''
        Use Greedy-search to generate predicted captions
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize,3,224,224)}
            dict of available visual features. when using AoASpatial model, only raw img tensors are required
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:sampled_ids:(bsize,max_len)
        '''
        images = visual_inputs['img_tensors']   #(bsize,3,224,224)
        enc_features = self.encoder(images) #(bsize,49,2048)
        projected_enc_features = self.img_feats_porjection(enc_features)    #(bsize,49,1024)
        refined_features = self.aoa_refine(x=projected_enc_features)
        sampled_ids,alphas = self.decoder.sample(refined_features,max_len=max_len)
        return sampled_ids

    def sampler_rl(self,visual_inputs,max_len=20):
        '''
        Use Monte Carlo method(Random Sampling) to generate sampled predictions for scst training
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize,3,224,224)}
            dict of available visual features. when using AoASpatial model, only raw img tensors are required
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:seq:(bsize,max_len)
                seqLogprobs:(bsize,max_len)
        '''
        images = visual_inputs['img_tensors']   #(bsize,3,224,224)
        enc_features = self.encoder(images) #(bsize,49,2048)
        projected_enc_features = self.img_feats_porjection(enc_features)    #(bsize,49,1024)
        refined_features = self.aoa_refine(x=projected_enc_features)
        seq,seqLogprob = self.decoder.sample_rl(refined_features,max_len=max_len)
        return seq,seqLogprob

    def beam_search_sampler(self,visual_inputs,beam_size=5):
        '''
        Use Beam search method to generate predicted captions, asserting bsize=1
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize,3,224,224)}
            dict of available visual features. when using AoASpatial model, only raw img tensors are required
        :param beam_size: beam numbers scalar
        :return: sampled_ids:(bsize=1,sentence_len)
        '''
        images = visual_inputs['img_tensors']   #(bsize,3,224,224)
        enc_features = self.encoder(images)
        projected_enc_features = self.img_feats_porjection(enc_features)    #(bsize,49,1024)
        refined_features = self.aoa_refine(x=projected_enc_features)
        sampled_ids,alphas = self.decoder.beam_search_sample(enc_features=refined_features,beam_size=beam_size)
        return sampled_ids

    def eval_test_image(self,visual_inputs,caption_vocab,max_len=20,eval_beam_size=-1):
        '''
        Tests on single given image.
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize=1,3,224,224)}
            dict of available visual features. when using AoASpatial model, only raw img tensors are required
        :param caption_vocab: pkl-file. used to translate the generated sentence into human language.
        :param max_len: maximum length(or time_step) of the generated sentence
        :param eval_beam_size: beam numbers scalar
        :return:caption: generated caption for given image.
                additional output: e.g. attention weights over different visual features at each time step during training.(not used in NIC)
        '''
        image = visual_inputs['img_tensors']    #(1,3,224,224)
        assert image.size(0) == 1
        enc_features = self.encoder(image)
        projected_enc_features = self.img_feats_porjection(enc_features)    #(bsize,49,1024)
        refined_features = self.aoa_refine(x=projected_enc_features)
        if eval_beam_size != -1:
            sampled_ids,alphas = self.decoder.beam_search_sample(enc_features=refined_features,beam_size=eval_beam_size)
        else:
            sampled_ids,alphas = self.decoder.sample(enc_features=refined_features,max_len=max_len)
        caption_ids = sampled_ids[0].cpu().detach().numpy()
        caption = []
        for word_id in caption_ids:
            word = caption_vocab.ix2word[word_id]
            if word == '<end>':
                break
            elif word != '<sta>':
                caption.append(word)
        return caption,[alphas]

#-------------------------------------------------------------------------------------#
#---doing bu_feature_projection with masks
def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, bu_feats, bu_masks):
    if bu_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(bu_feats, bu_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(bu_feats)

class AoADetection_Captioner(nn.Module):
    def __init__(self,vocab_size,num_heads=8,hidden_dim=1024,embed_dim=1024,dropout_aoa=0.3,dropout_prob=0.5,device='cpu'):
        super(AoADetection_Captioner,self).__init__()
        #img_embed_dim=d_model=hidden_dim=1024
        self.img_feats_porjection = nn.Sequential(
            nn.Linear(in_features=2048,out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.aoa_refine = AoA_Refine_Core(num_heads=num_heads,d_model=hidden_dim,dropout_aoa=dropout_aoa)
        self.decoder = AoA_Decoder(hidden_dim=hidden_dim,num_heads=num_heads,embed_dim=embed_dim,vocab_size=vocab_size,d_model=hidden_dim,dropout=dropout_prob,device=device)

    def get_param_groups(self,lr_dict):
        captioner_params = list(self.decoder.parameters())
        param_groups = [
            {'params':captioner_params,'lr':lr_dict['lr']}
        ]
        return param_groups

    def forward(self,visual_inputs,captions,lengths):
        '''
        :param visual_inputs: {'bu_feats':torch.FloatTensor,'bu_bboxes':[np.ndarry],'bu_masks':None/torch.FloatTensor}
            dict of available visual features. when using AoADetection model, both 'fixed' and 'adaptive' bottom_up features are supported.
            'fixed': bu_feats:(bsize,36,2048),bu_bboxes:(36,4),bu_masks: None
            'adaptive': bu_feats:(bsize,bu_len,2048),bu_bboxes:(bu_len,4),bu_masks:(bsize,bu_len)
                bu_len(10~100) is the maximum number of bu_features each image has in this batch
        :param captions: (bsize,max_len) sorted in length,torch.LongTensor
                [[1(<sta>),24,65,3633,54,234,67,34,12,2(<end>)],    #max_len=10
                [1(<sta>),45,5434,235,12,58,11,2(<end>),0,0],
                ...
                [1(<sta>),24,7534,523,12,2(<end>),0,0,0,0]]
        :param lengths:[9,7,...,5]
                #note that length[i] = len(caption[i])-1 since we skip the 2<end> token when feeding the captions-tensor into the model
                and we skip the 1<sta> token when generating predictions for loss calculation. Thus the total training step in this batch equals to max(lengths)
        :return:packed_padded_predictions: ((total_tokens_nums,vocab_size),indices(ignore))
        '''
        bottom_up_features = visual_inputs['bu_feats']
        bu_mask = visual_inputs['bu_masks']
        projected_enc_features = pack_wrapper(module=self.img_feats_porjection,bu_feats=bottom_up_features,bu_masks=bu_mask)    #(bsize,36/10~100,1024=d_model)
        refined_features = self.aoa_refine(x=projected_enc_features,bu_mask=bu_mask)
        packed_predictions = self.decoder(refined_features,captions,lengths,bu_mask)
        return packed_predictions

    def sampler(self,visual_inputs,max_len=20):
        '''
        Use Greedy-search to generate predicted captions
        :param visual_inputs: {'bu_feats':torch.FloatTensor,'bu_bboxes':[np.ndarry],'bu_masks':None/torch.FloatTensor}
            dict of available visual features. when using AoADetection model, both 'fixed' and 'adaptive' bottom_up features are supported.
            'fixed': bu_feats:(bsize,36,2048),bu_bboxes:(36,4),bu_masks: None
            'adaptive': bu_feats:(bsize,bu_len,2048),bu_bboxes:(bu_len,4),bu_masks:(bsize,bu_len)
                bu_len(10~100) is the maximum number of bu_features each image has in this batch
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:sampled_ids:(bsize,max_len)
        '''
        bottom_up_features = visual_inputs['bu_feats']
        bu_mask = visual_inputs['bu_masks']
        projected_enc_features = pack_wrapper(module=self.img_feats_porjection,bu_feats=bottom_up_features,bu_masks=bu_mask)    #(bsize,36/10~100,1024=d_model)
        refined_features = self.aoa_refine(x=projected_enc_features,bu_mask=bu_mask)
        sampled_ids,alphas = self.decoder.sample(enc_features=refined_features,max_len=max_len,bu_masks=bu_mask)
        return sampled_ids

    def sampler_rl(self,visual_inputs,max_len=20):
        '''
        Use Monte Carlo method(Random Sampling) to generate sampled predictions for scst training
        :param visual_inputs: {'bu_feats':torch.FloatTensor,'bu_bboxes':[np.ndarry],'bu_masks':None/torch.FloatTensor}
            dict of available visual features. when using AoADetection model, both 'fixed' and 'adaptive' bottom_up features are supported.
            'fixed': bu_feats:(bsize,36,2048),bu_bboxes:(36,4),bu_masks: None
            'adaptive': bu_feats:(bsize,bu_len,2048),bu_bboxes:(bu_len,4),bu_masks:(bsize,bu_len)
                bu_len(10~100) is the maximum number of bu_features each image has in this batch
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:seq:(bsize,max_len)
                seqLogprobs:(bsize,max_len)
        '''
        bottom_up_features = visual_inputs['bu_feats']
        bu_mask = visual_inputs['bu_masks']
        projected_enc_features = pack_wrapper(module=self.img_feats_porjection,bu_feats=bottom_up_features,bu_masks=bu_mask)    #(bsize,36/10~100,1024=d_model)
        refined_features = self.aoa_refine(x=projected_enc_features,bu_mask=bu_mask)
        seq,seqLogprob = self.decoder.sample_rl(refined_features,max_len=max_len,bu_masks=bu_mask)
        return seq,seqLogprob

    def beam_search_sampler(self,visual_inputs,beam_size=5):
        '''
        Use Beam search method to generate predicted captions, asserting bsize=1
        :param visual_inputs: {'bu_feats':torch.FloatTensor,'bu_bboxes':[np.ndarry],'bu_masks':None/torch.FloatTensor}
            dict of available visual features. when using AoADetection model, both 'fixed' and 'adaptive' bottom_up features are supported.
            'fixed': bu_feats:(bsize,36,2048),bu_bboxes:(36,4),bu_masks: None
            'adaptive': bu_feats:(bsize,bu_len,2048),bu_bboxes:(bu_len,4),bu_masks:(bsize,bu_len)
                bu_len(10~100) is the maximum number of bu_features each image has in this batch
        :param beam_size: beam numbers scalar
        :return: sampled_ids:(bsize=1,sentence_len)
        '''
        bottom_up_features = visual_inputs['bu_feats']
        bu_mask = visual_inputs['bu_masks']
        projected_enc_features = pack_wrapper(module=self.img_feats_porjection,bu_feats=bottom_up_features,bu_masks=bu_mask)    #(bsize,36/10~100,1024=d_model)
        refined_features = self.aoa_refine(x=projected_enc_features,bu_mask=bu_mask)
        sampled_ids,alphas = self.decoder.beam_search_sample(enc_features=refined_features,bu_masks=bu_mask,beam_size=beam_size)
        return sampled_ids

    def eval_test_image(self,visual_inputs,caption_vocab,max_len=20,eval_beam_size=-1):
        '''
        Tests on single given image.
        :param visual_inputs: {'bu_feats':torch.FloatTensor,'bu_bboxes':[np.ndarry],'bu_masks':None/torch.FloatTensor}
            dict of available visual features. when using AoADetection model, both 'fixed' and 'adaptive' bottom_up features are supported.
            'fixed': bu_feats:(bsize,36,2048),bu_bboxes:(36,4),bu_masks: None
            'adaptive': bu_feats:(bsize,bu_len,2048),bu_bboxes:(bu_len,4),bu_masks:(bsize,bu_len)
                bu_len(10~100) is the maximum number of bu_features each image has in this batch
        :param caption_vocab: pkl-file. used to translate the generated sentence into human language.
        :param max_len: maximum length(or time_step) of the generated sentence
        :param eval_beam_size: beam numbers scalar
        :return:caption: generated caption for given image.
                additional output: e.g. attention weights over different visual features at each time step during training.(not used in NIC)
        '''
        bottom_up_features = visual_inputs['bu_feats']
        bu_mask = visual_inputs['bu_masks']
        assert bottom_up_features.size(0) == 1
        projected_enc_features = pack_wrapper(module=self.img_feats_porjection,bu_feats=bottom_up_features,bu_masks=bu_mask)    #(bsize,36/10~100,1024=d_model)
        refined_features = self.aoa_refine(x=projected_enc_features,bu_mask=bu_mask)
        if eval_beam_size != -1:
            sampled_ids,alphas = self.decoder.beam_search_sample(enc_features=refined_features,beam_size=eval_beam_size,bu_masks=bu_mask)
        else:
            sampled_ids,alphas = self.decoder.sample(enc_features=refined_features,max_len=max_len,bu_masks=bu_mask)
        caption_ids = sampled_ids[0].cpu().detach().numpy()
        caption = []
        for word_id in caption_ids:
            word = caption_vocab.ix2word[word_id]
            if word == '<end>':
                break
            elif word != '<sta>':
                caption.append(word)
        return caption,[alphas]
