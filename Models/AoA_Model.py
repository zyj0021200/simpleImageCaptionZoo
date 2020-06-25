import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
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
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(feature_dim=size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

def DotProductAttention(query,key,value,mask=None,dropout=None):
    '''
    Compute 'Scaled Dot Product Attention'
    :param query:(bsize,num_heads,query_feature_nums,d_q)
    :param key:(bsize,num_heads,keyValue_feature_nums,d_k)
    :param value:(bsize,num_heads,keyValue_feature_nums,d_v)
    :param mask:(bsize,num_heads,query_feature_nums,keyValue_feature_nums)
    :param dropout:torch.nn.dropout()
    :return:value_:(bsize,num_heads,query_feature_nums,d_v)
            p_atten:(bsize,num_heads,query_feature_nums,keyValue_feature_nums)
    '''
    d_k = key.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)      #(bsize,num_heads,feature_nums,feature_nums)
    if mask is not None:
        scores = scores.masked_fill(mask==0,-1e9)
    p_atten = torch.softmax(scores,dim=-1)
    if dropout is not None:
        p_atten = dropout(p_atten)
    value_ = torch.matmul(p_atten,value)
    return value_,p_atten

class AoABlock(nn.Module):
    def __init__(self,num_heads,d_model,dropout_aoa=0.3,dropout=0.1):
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
        self.aoa_layer = nn.Sequential(nn.Linear(in_features=2*d_model,out_features=2*d_model),nn.GLU())
        if dropout_aoa>0:
            self.dropout_aoa = nn.Dropout(p=dropout_aoa)
        else:
            self.dropout_aoa = lambda x:x
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        '''
        Compute AoA^E(f_mh-att,Q,K,V)
        :param query:(bsize,query_feature_nums,d_model)
        :param key:(bsize,key_value_feature_nums,d_model)
        :param value:(bsize,key_value_feature_nums,d_model)
        :param mask:(bsize,query_feature_nums,key_value_feature_nums)
        :return:x:(bsize,query_feature_nums,d_model)
        '''
        bsize = query.size(0)
        # (bsize,feature_nums,d_model)->(bsize,feature_nums,d_model)->
        # ->(bsize,feature_nums,num_heads,d_q/k/v)->(bsize,num_heads,feature_nums,d_q/k/v) d_k=d_q=d_v
        query_p = self.linear_Q(query).view(bsize,-1,self.num_heads,self.d_q).transpose(1,2)
        key_p = self.linear_K(key).view(bsize, -1, self.num_heads, self.d_k).transpose(1, 2)
        value_p = self.linear_V(value).view(bsize, -1, self.num_heads, self.d_v).transpose(1, 2)
        x,atten = DotProductAttention(query=query_p,key=key_p,value=value_p,mask=mask,dropout=self.dropout) #(bsize,num_heads,query_feature_nums,d_v)
        x = x.transpose(1,2).contiguous().view(bsize,-1,self.num_heads*self.d_v)    #(bsize,query_feature_nums,d_model)
        x = self.aoa_layer(self.dropout_aoa(torch.cat([x,query],dim=-1)))     #(bsize,query_feature_nums,d_model)
        return x

class AoA_Refine_Core(nn.Module):
    def __init__(self, num_heads, d_model=1024, dropout_aoa=0.3,dropout=0.1):
        super(AoA_Refine_Core,self).__init__()
        self.aoa_block = AoABlock(
            num_heads=num_heads,
            d_model=d_model,
            dropout_aoa=dropout_aoa,
            dropout=dropout
        )
        self.sublayer = SublayerConnection(size=d_model,dropout=dropout)
        self.aoa_layers = clones(self.sublayer,N=6)
        self.norm = LayerNorm(feature_dim=d_model)

    def forward(self,x,mask=None):
        '''
        Encoder with AoA
        :param x: (bsize,query_feature_nums,d_model)
        :param mask:
        :return:
        '''
        for layer in self.aoa_layers:
            x = layer(x,lambda x:self.aoa_block(query=x,key=x,value=x,mask=mask))
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
        self.aoa_block = AoABlock(num_heads=num_heads,d_model=d_model)
        self.embed = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim)
        self.predict = weight_norm(nn.Linear(in_features=hidden_dim,out_features=vocab_size))
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1,0.1)
        self.predict.weight.data.uniform_(-0.1,0.1)
        self.predict.bias.data.fill_(0)

    def init_hidden_state(self,bsize):
        h = torch.zeros(bsize,self.hidden_dim).to(self.device)
        m = torch.zeros(bsize,self.hidden_dim).to(self.device)
        ctx = torch.zeros(bsize,self.hidden_dim).to(self.device)
        return h,m,ctx

    def forward(self,enc_features,captions,lengths,atten_masks=None):
        bsize = enc_features.size(0)
        num_features = enc_features.size(1)   #(bsize,49/36,1024=hidden_dim=d_model)
        mean_features = torch.mean(enc_features,dim=1,keepdim=False)    #(bsize,1024)
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
                torch.cat([embeddings,mean_features[:bsize_t]+ctx[:bsize_t]],dim=1),
                (h[:bsize_t],m[:bsize_t])
            )
            if atten_masks is not None:
                atten_masks_this_step = atten_masks[:bsize_t]
            else:
                atten_masks_this_step = None
            ctx = self.aoa_block(
                query=h.unsqueeze(1),
                key=enc_features[:bsize_t],
                value=enc_features[:bsize_t],
                mask=atten_masks_this_step
            ) #(bsize,1,d_model)
            ctx = ctx.squeeze(1)    #(bsize,d_model=hidden_dim)
            preds = self.predict(self.dropout(ctx))
            predictions[:bsize_t, time_step, :] = preds

        pack_predictions = pack_padded_sequence(predictions, lengths, batch_first=True)
        return pack_predictions

    def sample(self,enc_features,max_len=20,atten_masks=None):
        bsize = enc_features.size(0)
        num_features = enc_features.size(1)   #(bsize,49,2048)
        mean_features = torch.mean(enc_features,dim=1,keepdim=False)    #(bsize,1024)
        h,m,ctx = self.init_hidden_state(bsize)
        captions = torch.LongTensor(bsize,1).fill_(1).to(self.device)
        sampled_ids = []
        for time_step in range(max_len):
            embeddings = self.embed(captions).squeeze(1)    #(bsize,embed_dim)
            h,m = self.lstm(
                torch.cat([embeddings,mean_features+ctx],dim=1),
                (h,m)
            )
            if atten_masks is not None:
                atten_masks_this_step = atten_masks
            else:
                atten_masks_this_step = None
            ctx = self.aoa_block(
                query=h.unsqueeze(1),
                key=enc_features,
                value=enc_features,
                mask=atten_masks_this_step
            ) #(bsize,1,d_model)
            ctx = ctx.squeeze(1)    #(bsize,d_model=hidden_dim)
            preds = self.predict(self.dropout(ctx))
            pred_id = preds.max(1)[1]  # (bsize,)
            captions = pred_id.unsqueeze(1)  # (bsize,1)
            sampled_ids.append(captions)
        sampled_ids = torch.cat(sampled_ids, dim=1)  # (bsize,max_seq)
        return sampled_ids

    def beam_search_sample(self,enc_features,beam_size=5,atten_masks=None):
        '''
        :param enc_features:(1,h*w/36,1024)
        :param beam_size:scalar
        :return:
        '''
        num_features = enc_features.size(1)
        k = beam_size
        k_prev_words = torch.LongTensor(k,1).fill_(1).to(self.device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k,1).to(self.device)
        mean_features = torch.mean(enc_features,dim=1,keepdim=False)    #(1,1024)
        enc_features = enc_features.expand(k,enc_features.shape[1],enc_features.shape[2])   #(k,h*w/36,1024)
        mean_features = mean_features.expand(k,mean_features.shape[1])    #(k,1024)
        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        max_step_limit = 50
        h,m,ctx = self.init_hidden_state(bsize=beam_size)
        while step<=max_step_limit:
            embeddings = self.embed(k_prev_words).squeeze(1)    #(s=active_beam_num,embed_dim)

            h,m = self.lstm(
                torch.cat([embeddings,mean_features+ctx],dim=1),
                (h,m)
            )
            if atten_masks is not None:
                atten_masks_this_step = atten_masks
            else:
                atten_masks_this_step = None
            ctx = self.aoa_block(
                query=h.unsqueeze(1),
                key=enc_features,
                value=enc_features,
                mask=atten_masks_this_step
            )   #(s,1,d_model=hidden_dim=1024)
            ctx = ctx.squeeze(1)    #(s,d_model)
            scores = self.predict(self.dropout(ctx))
            scores = Fun.log_softmax(scores,dim=1)
            scores = top_k_scores.expand_as(scores) + scores    #(s,vocab_size)
            if step == 1:
                top_k_scores,top_k_words = scores[0].topk(k,0,True,True) #(s)
            else:
                top_k_scores,top_k_words = scores.view(-1).topk(k,0,True,True)
            prev_word_inds = top_k_words / self.vocab_size
            next_word_inds = top_k_words % self.vocab_size
            seqs = torch.cat([seqs[prev_word_inds],next_word_inds.unsqueeze(1)],dim=1)  #(s,step+1)

            incomplete_inds = [ind for ind,next_word in enumerate(next_word_inds) if next_word != 2]
            complete_inds = list(set(range(len(next_word_inds)))-set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

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
        else:
            i = torch.max(top_k_scores,dim=0)[1].item()
            seq_tensor = seqs[i].unsqueeze(0)

        return seq_tensor

#--------------------------------Model Class----------------------------------#
class AoASpatial_Captioner(nn.Module):
    def __init__(self,encoded_img_size,vocab_size,num_heads=8,hidden_dim=1024,embed_dim=1024,dropout_aoa=0.3,dropout=0.5,device='cpu'):
        super(AoASpatial_Captioner,self).__init__()
        #img_embed_dim=hidden_dim=1024
        self.encoder = EncoderCNN(encoded_img_size=encoded_img_size)
        self.img_feats_porjection = nn.Linear(in_features=2048,out_features=hidden_dim)
        self.aoa_refine = AoA_Refine_Core(num_heads=num_heads,d_model=hidden_dim,dropout_aoa=dropout_aoa)
        self.decoder = AoA_Decoder(hidden_dim=hidden_dim,num_heads=num_heads,embed_dim=embed_dim,vocab_size=vocab_size,d_model=hidden_dim,dropout=dropout,device=device)
        self.cnn_fine_tune(flag=False)

    def cnn_fine_tune(self, flag=False):
        if flag:
            for module in list(self.encoder.feature_extractor.children())[7:]:
                for param in module.parameters():
                    param.requires_grad = True
        else:
            for params in self.encoder.feature_extractor.parameters():
                params.requires_grad = False

    def forward(self,images,captions,lengths):
        enc_features = self.encoder(images) #(bsize,49,2048)
        projected_enc_features = self.img_feats_porjection(enc_features)    #(bsize,1024)
        refined_features = self.aoa_refine(x=projected_enc_features)
        packed_predictions = self.decoder(refined_features,captions,lengths)
        return packed_predictions

    def sampler(self,images,max_len=20):
        enc_features = self.encoder(images) #(bsize,49,2048)
        projected_enc_features = self.img_feats_porjection(enc_features)    #(bsize,49,1024)
        refined_features = self.aoa_refine(x=projected_enc_features)
        sampled_ids = self.decoder.sample(refined_features,max_len=max_len)
        return sampled_ids

    def beam_search_sampler(self,images,beam_size=5):
        enc_features = self.encoder(images)
        projected_enc_features = self.img_feats_porjection(enc_features)    #(bsize,49,1024)
        refined_features = self.aoa_refine(x=projected_enc_features)
        sampled_ids = self.decoder.beam_search_sample(enc_features=refined_features,beam_size=beam_size)
        return sampled_ids

#-------------------------------------------------------------------------------------#
class AoADetection_Captioner(nn.Module):
    def __init__(self,vocab_size,num_heads=8,hidden_dim=1024,embed_dim=1024,dropout_aoa=0.3,dropout=0.5,device='cpu'):
        super(AoADetection_Captioner,self).__init__()
        #img_embed_dim=d_model=hidden_dim=1024
        self.img_feats_porjection = nn.Linear(in_features=2048,out_features=hidden_dim)
        self.aoa_refine = AoA_Refine_Core(num_heads=num_heads,d_model=hidden_dim,dropout_aoa=dropout_aoa)
        self.decoder = AoA_Decoder(hidden_dim=hidden_dim,num_heads=num_heads,embed_dim=embed_dim,vocab_size=vocab_size,d_model=hidden_dim,dropout=dropout,device=device)

    def forward(self,bottom_up_features,captions,lengths):
        projected_enc_features = self.img_feats_porjection(bottom_up_features)    #(bsize,36,1024=d_model)
        refined_features = self.aoa_refine(x=projected_enc_features)
        packed_predictions = self.decoder(refined_features,captions,lengths)
        return packed_predictions

    def sampler(self,bottom_up_features,max_len=20):
        projected_enc_features = self.img_feats_porjection(bottom_up_features)    #(bsize,36,1024)
        refined_features = self.aoa_refine(x=projected_enc_features)
        sampled_ids = self.decoder.sample(refined_features,max_len=max_len)
        return sampled_ids

    def beam_search_sampler(self,bottom_up_features,beam_size=5):
        projected_enc_features = self.img_feats_porjection(bottom_up_features)    #(bsize,36,1024)
        refined_features = self.aoa_refine(x=projected_enc_features)
        sampled_ids = self.decoder.beam_search_sample(enc_features=refined_features,beam_size=beam_size)
        return sampled_ids
