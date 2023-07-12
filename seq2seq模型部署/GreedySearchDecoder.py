import torch.jit
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
device = torch.device("cpu")

MAX_LENGTH = 10  # Maximum sentence length

class GreedySearchDecoder(torch.jit.ScriptModule):
    def __init__(self,encoder,decoder,decoder_n_layers):
        super(GreedySearchDecoder,self).__init__()
        self.encoder = encoder
        self.decoder =decoder
        self._device = device
        self._SOS_token = SOS_token
        self._decoder_n_layers = decoder_n_layers
        constants = ['_device','_SOS_token','_decoder_n_layers']

    @torch.jit.script_method
    def forward(self,input_seq:torch.Tensor,input_length:torch.Tensor,max_length:int):
        #通过编码器转发输入
        encoder_outputs,encoder_hidden = self.encoder(input_seq,input_length)#  准备编码器模型转发输入
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]#   使用SOS_token初始化解码器输入
        decoder_input =torch.ones(1,1,device=self._device,dtype=torch.long)*self._SOS_token
        #初始化张量以将解码后的单词附加到
        all_tokens = torch.zeros([0],device=self._device,dtype=torch.long)
        all_scores = torch.zeros([0],device=self._device) #一次迭代地解码一个词令牌
        for _ in range(max_length):
            #正向通过解码器
            decoder_output,decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
            #获得最可能的单词标记及其softmax分数
            decoder_scores,decoder_input = torch.max(decoder_output,dim=1)# 记录令牌和分数
            all_tokens = torch.cat((all_tokens,decoder_input),dim=0)
            all_scores = torch.cat((all_scores,decoder_scores),dim=0)#准备当前令牌作为下一个解码器输入
            decoder_input = torch.unsqueeze(decoder_input,0)#   返回词令牌和分数的集合
            return all_tokens,all_scores

