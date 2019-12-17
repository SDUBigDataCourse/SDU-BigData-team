import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as dataset
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_
from Data import load_vocab, load_x

import numpy as np
import math
import os


def init_params(model, escape=None):
    param_num = 0
    for name, param in model.named_parameters():
        if escape is not None and escape in name:
            print('no_init', name, param.size())
            continue
        print('[init]', name, '[size]', param.size())
        s_ = 1
        for ax_ in param.size():
            s_ *= ax_
        param_num += s_
        if param.data.dim() > 1:
            xavier_uniform_(param.data)
    print('[total params]', param_num)


def new_tensor(array, requires_grad=False):
    tensor = torch.tensor(array, requires_grad=requires_grad)
    # if torch.cuda.is_available():
    #     tensor = tensor.cuda()
    return tensor


def create_emb_layer(emb_matrix, non_trainable=True):
    vocab_size, emb_size = emb_matrix.size()
    emb_layer = nn.Embedding(vocab_size, emb_size, padding_idx=0)
    emb_layer.load_state_dict({'weight': emb_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer


class AutoEncoder(nn.Module):
    # 训练自动编码机
    def __init__(self, encoder, decoder, sos=1, eos=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos = sos
        self.eos = eos

    def forward(self, x, tgt, *args, **kwargs):
        dec_inp = torch.cat([new_tensor([self.sos] * x.size(0), requires_grad=False).long().unsqueeze(1), x],
                            dim=1)
        dec_tgt = torch.cat([x, new_tensor([0] * x.size(0), requires_grad=False).long().unsqueeze(1)], dim=1)
        enc_out = self.encoder(x)
        dec_out = self.decoder(dec_inp, enc_out, dec_tgt.ne(0).detach())
        return F.cross_entropy(dec_out.view(-1, dec_out.size(-1)), dec_tgt.view(-1), ignore_index=0)


class Classifier(nn.Module):
    # 训练情感分类器
    def __init__(self, encoder, discriminator):
        super().__init__()
        self.encoder = encoder
        self.discriminator = discriminator

    def forward(self, x, tgt, *args, **kwargs):
        enc_out = self.encoder(x)
        out = self.discriminator(enc_out)
        return F.binary_cross_entropy(out.view(-1), tgt.float().view(-1))


class Generator(nn.Module):
    # 训练a>b, b>a两个生成器，希望可以骗过情感分类器，判别器并重建（三个loss）
    def __init__(self, encoder, ab, ba, sent_discriminator, fake_discriminator):
        super().__init__()
        self.encoder = encoder
        self.ab = ab
        self.ba = ba
        self.sent_discriminator = sent_discriminator
        self.fake_discriminator = fake_discriminator

    def forward(self, x, tgt, src='a', from_encoder=None, *args, **kwargs):
        enc_out = self.encoder(x).detach() if from_encoder is None else from_encoder.detach()
        if src == 'a':
            forward = self.ab
            backward = self.ba
        else:
            forward = self.ab
            backward = self.ba
        fake = forward(enc_out)
        sent_predict = self.sent_discriminator(fake)
        sent_loss = F.binary_cross_entropy(sent_predict.view(-1), tgt.float().view(-1))
        fake_predict = self.fake_discriminator(fake)
        fake_loss = F.binary_cross_entropy(fake_predict.view(-1), torch.tensor(1.).repeat(fake_predict.size(0)))
        rebuild = backward(fake)
        rebuild_loss = F.l1_loss(rebuild.view(-1), enc_out.view(-1))
        return sent_loss.unsqueeze(0), fake_loss.unsqueeze(0), rebuild_loss.unsqueeze(0)


class Discriminator(nn.Module):
    # 训练判别器，希望能分辨数据是否被生成器处理过
    def __init__(self, encoder, ab, ba, fake_discriminator):
        super().__init__()
        self.encoder = encoder
        self.ab = ab
        self.ba = ba
        self.fake_discriminator = fake_discriminator

    def forward(self, x, tgt, src='a', from_encoder=None, *args, **kwargs):
        enc_out = self.encoder(x).detach() if from_encoder is None else from_encoder.detach()
        if src == 'a':
            forward = self.ab
        else:
            forward = self.ab
        fake = forward(enc_out).detach()
        true_loss = F.binary_cross_entropy(self.fake_discriminator(enc_out).view(-1),
                                           torch.tensor(1.).repeat(fake.size(0)))
        fake_loss = F.binary_cross_entropy(self.fake_discriminator(fake).view(-1),
                                           torch.tensor(0.).repeat(fake.size(0)))
        return true_loss.unsqueeze(0), fake_loss.unsqueeze(0)


class WeakSupervision(nn.Module):
    # 使用弱监督训练自动编码机和生成器
    def __init__(self, encoder, ab, ba, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ab = ab
        self.ba = ba

    def forward(self, x, tgt, src='a', *args, **kwargs):
        if src == 'a':
            forward = self.ab
        else:
            forward = self.ab
        enc_out = self.encoder(x)
        fake = forward(enc_out)
        dec_inp = torch.cat([new_tensor([self.bos] * tgt.size(0), requires_grad=False).long().unsqueeze(1), tgt], dim=1)
        dec_tgt = torch.cat([tgt, new_tensor([self.eos] * tgt.size(0), requires_grad=False).long().unsqueeze(1)], dim=1)
        dec_out = self.decoder(dec_inp, fake)
        return F.cross_entropy(dec_out, dec_tgt)


class Trainer:
    def __init__(self, vocab_size, emb_size, d_model, nhead, num_layers, emb_matrix, multi_emb, hidden_size,
                 dim_feedforward):
        from EUST import Encoder, Decoder, SentDiscriminator, FakeDiscriminator, A2B, B2A
        self.vocab_size = vocab_size
        if emb_matrix is None:
            self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        else:
            self.embedding = create_emb_layer(emb_matrix)
        self.encoder = Encoder(vocab_size=vocab_size, emb_size=emb_size, d_model=d_model, nhead=nhead,
                               num_layers=num_layers, emb_layer=self.embedding, multi_emb=multi_emb,
                               dim_feedforward=dim_feedforward)
        self.decoder = Decoder(vocab_size=vocab_size, emb_size=emb_size, d_model=d_model, nhead=nhead,
                               num_layers=num_layers, emb_layer=self.embedding, dim_feedforward=dim_feedforward)
        self.sent_discriminator = SentDiscriminator(d_model=d_model, nhead=multi_emb, hidden_size=10)
        self.fake_discriminator = FakeDiscriminator(d_model=d_model, nhead=multi_emb, hidden_size=10)
        self.ab = A2B(d_model=d_model, nhead=multi_emb, hidden_size=hidden_size)
        self.ba = B2A(d_model=d_model, nhead=multi_emb, hidden_size=hidden_size)

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=3e-4)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=3e-4)
        self.sent_optimizer = optim.Adam(self.sent_discriminator.parameters(), lr=3e-4)
        self.fake_optimizer = optim.Adam(self.sent_discriminator.parameters(), lr=3e-4)
        self.ab_optimizer = optim.Adam(self.ab.parameters(), lr=3e-4)
        self.ba_optimizer = optim.Adam(self.ba.parameters(), lr=3e-4)

        self.AutoEncoder = AutoEncoder(self.encoder, self.decoder)
        self.Classifier = Classifier(self.encoder, self.sent_discriminator)
        self.Generator = Generator(self.encoder, self.ab, self.ba, self.sent_discriminator, self.fake_discriminator)
        self.Discriminator = Discriminator(self.encoder, self.ab, self.ba, self.fake_discriminator)
        self.WeakSupervision = WeakSupervision(self.encoder, self.ab, self.ba, self.decoder)

        self.Model = {'encoder': (self.encoder, self.encoder_optimizer),
                      'decoder': (self.decoder, self.decoder_optimizer),
                      'sent_discriminator': (self.sent_discriminator, self.sent_optimizer),
                      'fake_discriminator': (self.fake_discriminator, self.fake_optimizer),
                      'ab': (self.ab, self.ab_optimizer),
                      'ba': (self.ba, self.ba_optimizer)}

        self.Schedule = {'AutoEncoder': self.AutoEncoder,
                         'Classifier': self.Classifier,
                         'Generator': self.Generator,
                         'Discriminator': self.Discriminator,
                         'WeakSupervision': self.WeakSupervision}

        self.Optimize = {'AutoEncoder': ('encoder', 'decoder'),
                         'Classifier': ('encoder', 'sent_discriminator'),
                         'Generator': ('ab', 'ba'),
                         'Discriminator': ('fake_discriminator',),
                         'WeakSupervision': ('encoder', 'ab', 'ba', 'decoder')}
        self.init_params()

    def init_params(self):
        for model in self.Model:
            print('################')
            print('[init_params]', model)
            init_params(self.Model[model][0])
        print('################')

    def train_batch(self, x, tgt, src, enable, epoch):
        report = []
        for schedule in enable:
            optimizer = self.Optimize[schedule]
            for p in optimizer:
                self.Model[p][1].zero_grad()
            loss = self.Schedule[schedule](x, tgt, src=src)
            if isinstance(loss, tuple) or isinstance(loss, list):
                loss = torch.cat(loss, dim=-1).mean()
                closs = loss.cpu().item()
            else:
                loss = loss.mean()
                closs = loss.cpu().item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Schedule[schedule].parameters(), 2)
            for p in optimizer:
                self.Model[p][1].step()
            report.append('[epoch]{} [schedule]{} [loss]{}'.format(epoch, schedule, closs))
            print(report[-1])
        return report

    def train_epoch(self, x, tgt, batch_size, enable, epoch):
        data = dataset.TensorDataset(x, tgt)
        train_loader = dataset.DataLoader(data, batch_size=batch_size, shuffle=True)
        for j, data in enumerate(train_loader, 0):
            self.train_batch(data[0], data[1], src='a', enable=enable, epoch=epoch)

    def save_weights(self, path):
        for model in self.Model:
            torch.save(self.Model[model][0].state_dict(), path + '_' + model + '.pkl')

    def load_weights(self, path):
        for model in self.Model:
            self.Model[model][0].load_state_dict(torch.load(path + '_' + model + '.pkl'))


if __name__ == '__main__':
    # {'AutoEncoder': self.AutoEncoder,
    #  'Classifier': self.Classifier,
    #  'Generator': self.Generator,
    #  'Discriminator': self.Discriminator,
    #  'WeakSupervision': self.WeakSupervision}
    args = dict(vocab_size=22543, emb_size=256, d_model=256, nhead=8, num_layers=2, emb_matrix=None, multi_emb=1,
                hidden_size=10, dim_feedforward=2048)
    agent = Trainer(**args)
    # agent.load_weights('model/')
    vocab2id, id2vocab, id2freq = load_vocab('data/vocab', t=2)
    print(len(vocab2id))
    data, label = load_x('data/test.tsv', vocab2id)
    print(max([len(x) for x in data]))
    print(len(data))

    agent.train_epoch(torch.tensor([[1, 2, 3], [4, 5, 0], [3, 8, 6], [4, 4, 3]]),
                      torch.tensor([[1], [0], [1], [0]]),
                      batch_size=2, epoch=0,
                      enable=('AutoEncoder', 'Classifier', 'Generator', 'Discriminator',))
    # agent.save_weights('model/')
