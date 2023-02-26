import torch
from torch import Tensor,nn
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange
import os

import numpy as np
import matplotlib.pyplot as plt

from modules.SpecAugment import SpecAugmentation
from modules.PANNs import Cnn10
from modules.GAT import GraphAttentionLayer, GraphAttentionLayer_topk

NHEAD = 8
SOS_IDX = 0
PAD_IDX = 4367
EOS_IDX = 9

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a LayerNorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, topk=None):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        if not topk:  # if not topk, it is a standard GAT.
            self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        else:  # if topk, it is a GAT with the topk edges selection.
            self.attentions = [GraphAttentionLayer_topk(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, k_num=topk) for _ in range(nheads)]
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        '''
        input:
            x  : (B, T, D), T (time) is the nodes demention, D is the hidden dimension.
            adj: (B, T, T), A full connected adjacency matrix.
        output:
            x  : (T, B, D), We transpose the T dimension and B dimension as the input of decoder.
        '''
        pre_x = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        
        x = pre_x + x  # residual connection
        
        return x.transpose(0, 1)
    
    def plot(self, x, adj):
        '''
        input:
            x  : (B, T, D), T (time) is the nodes demention, D is the hidden dimension.
            adj: (B, T, T), A full connected adjacency matrix.
        output:
            graph_1  : (B, T, T), The graph of the attention layer.
            graph_2  : (B, T, T), The graph of the out_att layer.
        '''
        pre_x = x
        x = F.dropout(x, self.dropout, training=self.training)
        x_1 = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        graph_1 = torch.cat([att.plot(x, adj) for att in self.attentions], dim=-1)

        x = pre_x + x_1  # residual connection
        return graph_1

class GraphAC(nn.Module):
    def __init__(self, frequency_dim, hidden_dim, emb_size=256, dim_feedforward=512, dropout=.2,
                 num_decoder_layers=3, spec_aug=False, nb_classes=None, top_k=None):
        super().__init__()

        dict = torch.load('./modules/CNN10_encoder.pth')
        self.encoder = Cnn10(spec_aug=spec_aug)
        self.encoder.load_state_dict(dict)
        print(f'top_k: {top_k}')
        self.GAT = GAT(emb_size, emb_size, dropout, 0.2, nheads=1, topk=top_k)  # GAT_top25

        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, nb_classes)

        self.nb_classes = nb_classes

        self.tgt_tok_emb = TokenEmbedding(nb_classes, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.src_mask = None
        self.tgt_mask = None
        self.src_padding_mask = None
        self.tgt_padding_mask = None

    def forward(self, src: Tensor,  # (B, T, F)
                tgt: Tensor,  # (B, Seq_tgt)
                with_pad = False,
                mixup_param = None,
                ):
        if mixup_param and self.training:
            lam, index = mixup_param
            src = lam * src + (1-lam) * src[index]

        # The sequence without the last word as input. \
        # Its target output is the sequence without the first word (<sos>).
        trg = tgt.transpose(0, 1).contiguous()[:-1, :]   # (Seq_tgt, B)
        tgt_emb = self.tgt_tok_emb(trg)
        if self.training and mixup_param is not None:
            lam, index = mixup_param
            tgt_emb = lam * tgt_emb + (1 - lam) * tgt_emb[:, index]
        tgt_emb = self.positional_encoding(tgt_emb)

        # get the mask
        self.tgt_mask, self.tgt_padding_mask = create_tgt_mask(trg, with_pad)

        # src: (B, T, F) -> (B, T', F'), F'=4*F; -> memory: (T', B, F') for decoder
        memory = self.encoder(src).transpose(0, 1)  # -> (B, T', F') for GAT, using T' as nodes dimension.
        memory = self.GAT(memory, torch.ones(memory.shape[0], memory.shape[1], memory.shape[1]).to(memory.device))  # -> memory: (T', B, F') for decoder

        outs = self.transformer_decoder(tgt_emb, memory, self.tgt_mask, None,
                                        self.tgt_padding_mask, self.src_padding_mask)
        return self.generator(outs).transpose(0, 1).contiguous()

    def greedy_decode(self, src, max_steps=22, start_symbol=0):
        device = src.device
        src = src.transpose(0, 1)   # (Seq, B)
        if self.src_mask is None or self.src_mask.shape[0] != src.shape[0]:
            self.src_mask, self.src_padding_mask = create_src_mask(src)
        memory = self.encode(src,self.src_mask)
        ys = torch.ones(1,src.shape[1]).fill_(start_symbol).type(torch.long).to(device)
        for i in range(max_steps):
            memory = memory.to(device)
            tgt_mask = (generate_square_subsequent_mask(ys.shape[0], device)
                        .type(torch.bool)).to(device)

            out = self.decode(ys, memory, tgt_mask)
            prob = self.generator(out[-1,:])
            _, next_word = torch.max(prob, dim=1)

            ys = torch.cat([ys, next_word.unsqueeze(0)],dim=0)
        ys = nn.functional.one_hot(ys.transpose(0,1),self.nb_classes).transpose(0,1).float()
        return ys[1:, :, :].transpose()

    def init_vars(self, src, k_beam, max_steps=22, with_pad=False):
        device = src.device
        memory = self.encode(src)  # (Seq, B:1, hid)
        outputs = torch.LongTensor([[SOS_IDX]]).to(device)

        tgt_mask = (generate_square_subsequent_mask(1, device)  # tgt_mask may be checked carefully in the future
                    .type(torch.bool)).to(device)
        out = self.generator(self.decode(outputs, memory, tgt_mask))
        out = F.softmax(out, dim=-1).transpose(0, 1).contiguous()  # (Seq, B:1, nb_classes) -> (B:1, Seq, nb_classes)
        if with_pad:
            out[:, :, -1] = 0   # ignore <pad>

        probs, ix = out[:, -1].topk(k_beam)  # (B:1, nb_classes) -> (B:1, k_beam)
        log_scores = torch.log(probs)

        outputs = torch.zeros(k_beam, max_steps).long().to(device)  # (Seq/nb_classes, k_beam)
        outputs[:, 0] = SOS_IDX
        outputs[:, 1] = ix[0]  # k_beam words of ix are the candidates.

        # Check the memory is necessary!
        k_memorys = torch.zeros(memory.shape[0], k_beam, memory.shape[-1]).to(device)
        k_memorys[:, :] = memory[:, :]  # expand memory to k numbers
        k_memorys = k_memorys#.transpose(0, 1)  # (Seq, k_beam, hid)

        return outputs, k_memorys, log_scores

    def beam_search(self, src, k_beam=5, max_steps=22, with_pad=False):
        # to do init
        outputs, k_memorys, log_scores = self.init_vars(src, k_beam=k_beam, with_pad=with_pad)

        device = src.device
        if self.src_mask is None or self.src_mask.shape[0] != src.shape[0]:
            self.src_mask, self.src_padding_mask = create_src_mask(src)
        ind = None  # an important variable to decide the final output sequence
        EOS_check = torch.zeros(k_beam).bool()
        for i in range(2, max_steps):
            tgt_mask = (generate_square_subsequent_mask(i, device)  # tgt_mask may be checked carefully in the future
                        .type(torch.bool)).to(device)

            out = self.generator(self.decode(outputs[:, :i].transpose(0, 1).contiguous(), k_memorys, tgt_mask))
            out = F.softmax(out, dim=-1).transpose(0, 1).contiguous()  # (Seq, k_beam, nb_classes) -> (k_beam, Seq, nb_classes)
            if with_pad:
                out[:, :, -1] = 0  # ignore <pad>

            outputs, log_scores, EOS_check = self.k_best_outputs(outputs, out, log_scores, i, k_beam, EOS_check)

            ones = (outputs == EOS_IDX).nonzero()  # Occurrences of end symbols for all input summaries.
            summary_lengths = torch.zeros(len(outputs), dtype=torch.long).to(device)

            for vec in ones:
                i = vec[0]
                EOS_check[i] = True
                if summary_lengths[i] == 0:  # First end symbol has not been found yet
                    summary_lengths[i] = vec[1]  # Position of first end symbol

            num_finished_summaries = len([s for s in summary_lengths if s > 0])

            if num_finished_summaries == k_beam:
                alpha = 0.7
                div = 1 / (summary_lengths.type_as(log_scores) ** alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break

        if ind is None:
            _, ind = torch.max(log_scores, 1)
            ind = ind.data[0]
            ys = outputs[ind][1:]
        else:
            ys = outputs[ind][1:]
        ys = F.one_hot(ys.unsqueeze(0), self.nb_classes).float()

        return ys

    def k_best_outputs(self, outputs, out, log_scores, i, k_beam, EOS_check):
        device = log_scores.device
        probs, ix = out[:, -1].topk(k_beam)
        probs[EOS_check] = 1

        log_probs = torch.log(probs).to(device) \
                    + log_scores.transpose(0, 1).contiguous()
        k_probs, k_ix = log_probs.view(-1).topk(k_beam)
        row = k_ix // k_beam
        col = k_ix % k_beam

        outputs[:, :i] = outputs[row, :i]
        outputs[:, i] = ix[row, col]
        EOS_check = EOS_check[row]

        log_scores = k_probs.unsqueeze(0)
        return outputs, log_scores, EOS_check

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src: Tensor):
        # src: (B, T, F) -> (B, T', F'), F'=4*F; -> memory: (T', B, F') for decoder
        memory = self.encoder(src).transpose(0, 1)  # -> (B, T', F') for GAT, using T' as nodes dimension.
        memory = self.GAT(memory, torch.ones(memory.shape[0], memory.shape[1], memory.shape[1]).to(
            memory.device))  # -> memory: (T', B, F') for decoder
        return memory

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)
    
    def plot(self, src: Tensor, file_name: str, dir_path: str):
        memory = self.encoder(src).transpose(0, 1)  # -> (B, T', F') for GAT, using T' as nodes dimension.
        graphs = self.GAT.plot(memory, torch.ones(memory.shape[0], memory.shape[1], memory.shape[1]).to(memory.device))  # -> memory: (T', B, F') for decoder
        
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))
        
        graph = graphs[0]
        print(f'semantic association graph shape: {graph.shape}')
        im = axes.imshow(graph.squeeze(0).cpu(), vmin=0, vmax=1)
        axes.invert_yaxis()  # y轴反向
            
        axins1 = inset_axes(axes,
                    width="5%",  # width = 10% of parent_bbox width
                    height="100%",  # height : 50%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0., 1, 1),
                    bbox_transform=axes.transAxes,
                    borderpad=0,
                    )
        fig.colorbar(im, cax=axins1) 

        if not os.path.exists(dir_path): os.makedirs(dir_path)
        fig_path = os.path.join(dir_path, f'{file_name[:-4]}.svg')
        
        plt.savefig(fig_path)
        # plt.show()

        plt.figure(figsize=(20, 17))
        plt.imshow(memory.squeeze(0).transpose(0, 1).cpu() / torch.max(memory.cpu()), vmin=0, vmax=1)
        ax = plt.gca()
        ax.invert_yaxis()  # y轴反向
        plt.savefig(os.path.join(dir_path, f'{file_name[:-4]}_PANNs_feature.png'))
        
        return graph.squeeze(0).cpu().data.numpy()  # return the semantic associaiton graph for bilinear interpolation. 
        

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2) # tensor: (maxlen, 1, emb_size)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding) # register a buffer for pos_embedding.

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        # print(tokens)
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size) # why "* math.sqrt(self.emb_size)"?

def generate_square_subsequent_mask(sz,device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1).contiguous()
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    device = src.device
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len,device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    # src_padding_mask = (src == PAD_IDX).transpose(0, 1) # not fit the audio mel src
    src_padding_mask = None
    print(PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1).contiguous()

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def create_src_mask(src):
    device = src.device
    src_seq_len = src.shape[0]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    src_padding_mask = None

    return src_mask, src_padding_mask

def create_tgt_mask(tgt, with_pad=False):
    device = tgt.device
    tgt_seq_len = tgt.shape[0]

    # print(PAD_IDX)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len,device)
    if with_pad:
        tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1).contiguous()
    else:
        tgt_padding_mask = (tgt == EOS_IDX)
        index = (tgt_padding_mask == False).int().sum(dim=0, keepdim=False)
        for i in range(tgt.shape[1]):
            if index[i] < tgt.shape[0]:
                tgt_padding_mask[index[i], i] = 0
            else:
                pass
        tgt_padding_mask = tgt_padding_mask.transpose(0, 1).contiguous()

    return tgt_mask, tgt_padding_mask

if __name__=='__main__':

    model = GraphAC(frequency_dim=64, hidden_dim=256, emb_size=128, dim_feedforward=512, dropout=.2,
                 num_decoder_layers=3, spec_aug=False, nb_classes=4368)
    #
    print('Total amount of parameters: ',
                      f'{sum([i.numel() for i in model.encoder.parameters()])}')
    x = torch.ones(2, 2548, 64)
    y = model(x)