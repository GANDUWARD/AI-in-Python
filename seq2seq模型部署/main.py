from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

import EncoderRNN
import GreedySearchDecoder
import Voc
import LuoAttnDecoderRNN as L

# 默认的词向量
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
device = torch.device("cpu")

MAX_LENGTH = 10  # Maximum sentence length


# 小写并删除非字母字符
def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# 使用字符串句子，返回单词索引的句子
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # 格式化输入句子作为批处理
    # words->indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    #   创建长度张量
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转置批量的维度以匹配模型的期望
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 使用适当的设备
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # 用searcher解码句子s
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes ->words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


# 评估来自用户输入的输入（stdin）
def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while (1):
        try:
            #  获取输入的句子
            input_sentence = input('> ')
            # 检查是否是退出情况
            if input_sentence == 'q' or input_sentence == 'quit': break
            # 规范化句子
            input_sentence = normalizeString(input_sentence)
            # 评估句子
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 格式化和打印回复句
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))
        except KeyError:
            print("Error: Encountered unknown word.")


# 规范化输入句子并调用evaluate（）
def evaluateExample(sentence, encoder, decoder, searcher, voc):
    print(">" + sentence)
    # 规范化句子
    input_sentence = normalizeString(sentence)
    # 评估句子
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('Bot:', ' '.join(output_words))


if __name__ == '__main__':
    save_dir = os.path.join("data", "save")
    corpus_name = "cornell mocie-dialogs corpus"
    # 配置模型
    model_name = 'cb_model'
    attn_model = 'dot'
    # attn_model = 'general'
    # attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64
    # 如果你加载的是自己的模型
    # 设置要加载的检查点
    checkpoint_iter = 4000
    # loadFilename =os.path.join(save_dir,model_name,corpus_name,'{}-{}_{}'.format(encoder_n_layers,
    # decoder_n_layers,hidden_size),'{}_checkpoint.tar'.format(checkpoint_iter))

    # 如果你加载的是托管模型
    loadFilename = 'data/4000_checkpoint.tar'
    # 加载模型
    # 强制CPU设备选项
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc = Voc.Voc(corpus_name)
    voc.__dict__ = checkpoint['voc_dict']
    print('Building encoder and decoder ...')
    # 初始化词向量
    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    # 初始化编码器和解码器模型
    encoder = EncoderRNN.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = L.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    #加载训练模型参数
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    #使用适当的设备
    encoder =encoder.to(device)
    decoder = decoder.to(device)
    #将dropout层设置为eval模式
    encoder.eval()
    decoder.eval()
    print('Models built and ready to go!')
    #转换编码器模型
    #创建人工输入
    test_seq = torch.LongTensor(MAX_LENGTH,1).random_(0,voc.num_words).to(device)
    test_seq_length = torch.LongTensor([test_seq.size()[0]]).to(device)
    #跟踪模型
    traced_encoder = torch.jit.trace(encoder,(test_seq,test_seq_length))

    #转换解码器模型
    #创建并生成人工输入
    test_encoder_outputs ,test_encoder_hidden = traced_encoder(test_seq,test_seq_length)
    test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
    test_decoder_input =torch.LongTensor(1,1).random_(0,voc.num_words)
    #跟踪模型
    traced_decoder = torch.jit.trace(decoder,(test_decoder_input,
                                              test_decoder_hidden,test_encoder_outputs))
    #初始化searcher模块
    scripted_searcher = GreedySearchDecoder.GreedySearchDecoder(traced_encoder,traced_decoder,decoder_n_layers)
    #评估例子
    sentences =["hello","what's up?","who are you?","where am I?","where are you from?"]
    for s in sentences:
        evaluateExample(s,traced_encoder,traced_decoder,scripted_searcher,voc)
    #评估输入
    evaluateInput(traced_encoder,traced_decoder,scripted_searcher,voc)
