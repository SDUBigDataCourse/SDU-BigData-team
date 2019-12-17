import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

import numpy as np
import math

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


def neginf(dtype):
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def new_tensor(array, requires_grad=False):
    tensor = torch.tensor(array, requires_grad=requires_grad)
    # if torch.cuda.is_available():
    #     tensor = tensor.cuda()
    return tensor


def universal_sentence_embedding(sentences, mask, sqrt=True):
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = (mask.sum(dim=1).view(-1, 1).float())
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, neginf(torch.float32)).masked_fill(mask == 1, float(0.0))
    # if torch.cuda.is_available():
    #     mask = mask.cuda()
    return mask


def create_emb_layer(emb_matrix, non_trainable=True):
    vocab_size, emb_size = emb_matrix.size()
    emb_layer = nn.Embedding(vocab_size, emb_size, padding_idx=0)
    emb_layer.load_state_dict({'weight': emb_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        p = self.pe[:x.size(-2)]
        for i in range(len(x.size()) - 2):
            p = p.unsqueeze(0)
        x = x + p
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, d_model, nhead=1, num_layers=1, emb_layer=None, multi_emb=2,
                 dim_feedforward=2018):
        super().__init__()
        self.multi_emb = multi_emb
        self.embedding = emb_layer
        self.emb_dropout = nn.Dropout(0.1)
        self.pos_embedding = PositionalEmbedding(emb_size)
        self.utterEnc = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers,
            norm=LayerNorm(d_model))
        self.ext = nn.Linear(d_model, d_model * multi_emb)
        self.multi_pre = nn.TransformerEncoderLayer(d_model * multi_emb, multi_emb, dim_feedforward=dim_feedforward,
                                                    dropout=0.1,
                                                    activation='relu')

    def forward(self, x):
        mask = x.ne(0).detach()
        batch_size = x.size(0)
        out = self.pos_embedding(self.emb_dropout(self.embedding(x)))
        out = self.utterEnc(out.transpose(0, 1), src_key_padding_mask=~mask).transpose(0, 1)
        out = self.ext(out)
        out = self.multi_pre(out.transpose(0, 1), src_key_padding_mask=~mask).transpose(0, 1)
        pres = universal_sentence_embedding(out, mask)
        pres = pres.view(batch_size, self.multi_emb, -1)
        return pres


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, d_model, nhead=1, num_layers=1, emb_layer=None, dim_feedforward=2018):
        super().__init__()
        self.embedding = emb_layer
        self.emb_dropout = nn.Dropout(0.1)
        self.pos_embedding = PositionalEmbedding(emb_size)
        self.utterDec = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers,
            norm=LayerNorm(d_model))
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, mask):
        tgt = self.pos_embedding(self.emb_dropout(self.embedding(tgt)))

        tgt_out = self.utterDec(tgt.transpose(0, 1), memory.transpose(0, 1),
                                tgt_mask=_generate_square_subsequent_mask(tgt.size(1)),
                                tgt_key_padding_mask=~mask).transpose(0, 1)
        tgt_out = self.fc(tgt_out)
        return tgt_out

    def inference(self, memory, bos):
        N, L, S = memory.size()
        decoder_input = new_tensor([bos] * batch_size, requires_grad=False)
        all_decode_outputs = [dict({'state': decoder_states})]

        greedy_indices = list()
        greedy_end = new_tensor([0] * batch_size).long() == 1
        for t in range(max_len):
            decode_outputs = model.decode(
                data, decoder_input, encode_outputs, all_decode_outputs[-1]
            )

            gen_output = model.generate(data, encode_outputs, decode_outputs, softmax=True)

            probs, ids = model.to_word(data, gen_output, 1)

            all_decode_outputs.append(decode_outputs)

            greedy_indice = ids[:, 0]
            greedy_this_end = greedy_indice == EOS
            if t == 0:
                greedy_indice.masked_fill_(greedy_this_end, UNK)
            else:
                greedy_indice.masked_fill_(greedy_end, PAD)
            greedy_indices.append(greedy_indice.unsqueeze(1))
            greedy_end = greedy_end | greedy_this_end

            decoder_input = model.generation_to_decoder_input(data, greedy_indice)

        greedy_indice = torch.cat(greedy_indices, dim=1)
        return greedy_indice




class SentDiscriminator(nn.Module):
    def __init__(self, d_model, nhead, hidden_size):
        super().__init__()
        self.hidden = nn.Linear(d_model * nhead, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.hidden(x)
        out = self.fc(out)
        out = self.activation(out)
        return out


class FakeDiscriminator(nn.Module):
    def __init__(self, d_model, nhead, hidden_size):
        super().__init__()
        self.hidden = nn.Linear(d_model * nhead, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.hidden(x)
        out = self.fc(out)
        out = self.activation(out)
        return out


class A2B(nn.Module):
    def __init__(self, d_model, nhead, hidden_size):
        super().__init__()
        self.enc = nn.Linear(d_model * nhead, hidden_size)
        self.dec = nn.Linear(hidden_size, d_model * nhead)

    def forward(self, x):
        N, H, L = x.size()
        x = x.view(N, -1)
        out = self.enc(x)
        out = self.dec(out)
        return (x + out).view(N, H, L)


class B2A(nn.Module):
    def __init__(self, d_model, nhead, hidden_size):
        super().__init__()
        self.enc = nn.Linear(d_model * nhead, hidden_size)
        self.dec = nn.Linear(hidden_size, d_model * nhead)

    def forward(self, x):
        N, H, L = x.size()
        x = x.view(x.size(0), -1)
        out = self.enc(x)
        out = self.dec(out)
        return (x + out).view(N, H, L)


if __name__ == '__main__':
    emb = nn.Embedding(10, 4, padding_idx=0)
    ec = Encoder(vocab_size=10, emb_size=4, d_model=4, nhead=1, num_layers=1, emb_layer=emb, multi_emb=2)
    dc = Decoder(vocab_size=10, emb_size=4, d_model=4, nhead=1, num_layers=1, emb_layer=emb)
    sc = SentDiscriminator(d_model=4, nhead=2, hidden_size=10)
    fk = FakeDiscriminator(d_model=4, nhead=2, hidden_size=10)
    ab = A2B(d_model=4, nhead=2, hidden_size=10)
    ba = B2A(d_model=4, nhead=2, hidden_size=10)

    tmp = ec(torch.tensor([[1, 2, 3], [4, 5, 0]]))
    res = dc(torch.tensor([[1, 2, 3, 4], [4, 5, 0, 0]]), tmp)
    cls = sc(tmp)
    trs = ab(tmp)

    print(tmp.size(), res.size(), cls.size(), trs.size())
