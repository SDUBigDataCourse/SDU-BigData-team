import os
import codecs
import pickle
import torch
import numpy as np
import jieba
import sys
import time

PAD_WORD = '[PAD]'
BOS_WORD = '[BOS]'
UNK_WORD = '[UNK]'
EOS_WORD = '[EOS]'
SEP_WORD = '[SEP]'
CLS_WORD = '[CLS]'
MASK_WORD = '[MASK]'


def all_path(dirname):
    """
    获得文件夹下所有文件列表
    """
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result


def read_file(filename, vocab_dict, vocab_freq):
    """
    读文件添加到词表里
    filename：文件名
    vocab_dict：词典
    vocab_freq：词频
    """
    with open(filename, "r", encoding='utf-8') as f:
        for line in f:
            for word in jieba.cut(line.split('\t')[1]):
                if word in vocab_dict:
                    vocab_freq[vocab_dict[word]] += 1
                else:
                    vocab_dict[word] = vocab_dict['[INDEX]']
                    vocab_dict['[INDEX]'] += 1
                    vocab_freq.append(1)
    return vocab_dict, vocab_freq


def build_vocab(dirname, save_path):
    """
    构建词典和词频
    dirname：语料库的文件夹
    save_path：保存词典，按词频顺序
    """
    vocab_dict = {'[INDEX]': 0}
    vocab_freq = []
    path_list = all_path(dirname)
    for filename in path_list:
        vocab_dict, vocab_freq = read_file(filename, vocab_dict, vocab_freq)
    STOP_WORD = ['[INDEX]', 'NaN', '&lt;', 'SEP', '&gt;']
    vocab_freq.append(0)
    vocab_list = list(vocab_dict.keys())
    vocab_list.sort(key=lambda x: vocab_freq[vocab_dict[x]], reverse=True)
    vocab_freq.sort(reverse=True)
    with open(save_path, 'w') as wf:
        for word, freq in zip(vocab_list, vocab_freq):
            if word not in STOP_WORD:
                wf.write(word + '\t' + str(freq) + '\n')


def load_vocab(vocab_file, t=0, vocab_size=None):
    """
    从词频字典加载字典
    vocab_file：词频字典
    t：定义最小词频
    vocab_size：词典大小
    """
    thisvocab2id = {PAD_WORD: 0, BOS_WORD: 1, UNK_WORD: 2, EOS_WORD: 3, SEP_WORD: 4, CLS_WORD: 5, MASK_WORD: 6}
    thisid2vocab = [PAD_WORD, BOS_WORD, UNK_WORD, EOS_WORD, SEP_WORD, CLS_WORD, MASK_WORD]
    id2freq = [0 for _ in range(7)]
    with codecs.open(vocab_file, 'r') as f:
        for line in f:
            try:
                name, freq = line.strip('\n').strip('\r').split('\t')
            except:
                continue
            if int(freq) >= t:
                idx = len(thisid2vocab)
                thisvocab2id[name] = idx
                thisid2vocab.append(name)
                id2freq.append(int(freq))
            if vocab_size is not None and len(thisid2vocab) == vocab_size:
                break
    id2freq[0] = sum(id2freq) // len(id2freq)
    id2freq[1] = id2freq[0]
    id2freq[2] = id2freq[0]
    id2freq[3] = id2freq[0]

    print('item size: ', len(thisvocab2id))

    return thisvocab2id, thisid2vocab, id2freq


def load_x(filename, vocab2idx):
    """
    加载文本数据集
    filename：数据集文件
    vocab2id：vocab2id词典
    """
    if os.path.exists(filename + '.pkl'):
        with open(filename + '.pkl', 'rb') as f:
            return pickle.load(f)
    data = []
    label = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            m, line = line.split('\t')
            label.append(int(m))
            data.append([vocab2idx.get(item, 2) for item in jieba.cut(line)] + [3])
            # data.append([vocab2id.get(x, vocab2id[UNK_WORD]) for x in line.split(' ')] + [vocab2id[EOS_WORD]])
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump((data, label), f)
    return data, label


def load_embeddings(emb_text_filepath, vocab2idx, emb_dim):
    """
    加载预训练词向量
    emb_text_filepath：词向量文件
    vocab2idx：vocab2idx词典
    emb_dim：词向量大小
    """
    if os.path.exists(emb_text_filepath + '.pkl'):
        with open(emb_text_filepath + '.pkl', 'rb') as f:
            return pickle.load(f)

    matrix_len = len(vocab2idx)
    emb_matrix = torch.zeros((matrix_len, emb_dim))
    matched = [0 for _ in range(matrix_len)]
    with open(emb_text_filepath, "r", encoding='utf-8') as f:
        i = 0
        for line in f:
            tmp = line.strip().split(' ')
            if len(tmp) < 300:
                continue
            name = ' '.join(tmp[:-300])
            if name in vocab2idx:
                emb_matrix[vocab2idx[name]] = torch.tensor([float(x) for x in tmp[-300:]])
                matched[vocab2idx[name]] = 1
            if i % 20000 == 0:
                print(i, sum(matched), '/', matrix_len)
            i += 1
    for i in range(matrix_len):
        if matched[i] != 1:
            emb_matrix[i] = torch.tensor(np.random.normal(scale=0.6, size=(emb_dim,)))
    with open(emb_text_filepath + '.pkl', 'wb') as f:
        pickle.dump(emb_matrix, f)
    return emb_matrix


if __name__ == '__main__':
    # from transformers import BertTokenizer
    #
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # x, y = load_x('data/train.tsv', tokenizer)
    # build_vocab('data/s', 'data/vocab')
    vocab2id, id2vocab, id2freq = load_vocab('data/vocab', t=2)
    print(len(vocab2id))
    data, label = load_x('data/test.tsv', vocab2id)
    print(max([len(x) for x in data]))
    print(len(data))
