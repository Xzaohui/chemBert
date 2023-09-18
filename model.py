import logging
import math
import os
import torch.nn.functional as F
from transformers import BertConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
# from torch.nn.functional import gelu
import numpy as np

from transformers.activations import gelu,gelu_new
# from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer


class chemBert_c(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder=encoder
        self.line=nn.Linear(768,2)
    def forward(self,input_ids,attention_mask):
        out=self.encoder(input_ids,attention_mask).pooler_output
        out=self.line(out)
        return out
    
class chemBert_r(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder=encoder
        self.line=nn.Linear(768,1)
    def forward(self,input_ids,attention_mask):
        out=self.encoder(input_ids,attention_mask).pooler_output
        out=self.line(out)
        return out


class lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.enbedding=nn.Embedding(9999,512)
        self.lstm=nn.LSTM(512,512,2)
        self.line=nn.Linear(512,2)
    def forward(self,input_ids):
        out=self.enbedding(input_ids)
        out,_=self.lstm(out)
        out=self.line(out)
        return out[:,-1,:]


# class BertConfig(object):
#   """BERT模型的配置类."""
 
#   def __init__(self,
#                vocab_size=9999,
#                hidden_size=768,
#                num_hidden_layers=12,
#                num_attention_heads=12,
#                intermediate_size=3072,
#                hidden_dropout_prob=0.1,
#                attention_probs_dropout_prob=0.1,
#                max_position_embeddings=512,
#                type_vocab_size=16,
#                initializer_range=0.02,
#                hidden_act="gelu"):
 
#     self.vocab_size = vocab_size
#     self.hidden_size = hidden_size
#     self.num_hidden_layers = num_hidden_layers
#     self.num_attention_heads = num_attention_heads
#     self.pad_token_id=1
#     self.layer_norm_eps=1e-12
#     self.intermediate_size = intermediate_size
#     self.hidden_dropout_prob = hidden_dropout_prob
#     self.attention_probs_dropout_prob = attention_probs_dropout_prob
#     self.max_position_embeddings = max_position_embeddings
#     self.type_vocab_size = type_vocab_size
#     self.initializer_range = initializer_range
#     self.output_attentions = False
#     self.hidden_act=hidden_act
#     self.is_decoder = False
#     self.output_hidden_states = True

# config=BertConfig()

#mish激活函数
def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))
#一共5种激活函数
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": nn.functional.silu, "gelu_new": gelu_new, "mish": mish}


BertLayerNorm = torch.nn.LayerNorm #使用LayerNorm


class BertEmbeddings(nn.Module):
    """构造词嵌入特征，位置编码，token词类型的段落嵌入
    """

    def __init__(self, config):
        super().__init__()
        #词嵌入特征，位置编码，token词类型的段落嵌入
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        '''
        :param input_ids: 维度 [batch_size, sequence_length]，批次大小和句子长度
        :param positional_ids: 位置编码 [sequence_length, embedding_dimension]，句子长度，嵌入维度
        :param token_type_ids: BERT训练的时候, 会输入两个句子，第一句是0, 第二句是1
        :return: 嵌入的结果embedding，维度 [batch_size, sequence_length, embedding_dimension]
        '''
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1] #第1维度是句子长度
        #是否gpu
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        #最后的embedding结果是输入的词嵌入+位置嵌入+类型嵌入
        embeddings = inputs_embeds + position_embeddings #+ token_type_embeddings
        embeddings = self.LayerNorm(embeddings) #LN，加快收敛
        embeddings = self.dropout(embeddings) #dropout
        return embeddings #返回嵌入的结果
    

class BertSelfAttention(nn.Module):
    '''
    bert的自注意力机制，输入和输出的维度一致，为了残差的连接维度要一致
    '''
    def __init__(self, config):
        super().__init__()
        # 判断embedding dimension是否可以被num_attention_heads整除
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads #注意力的头数
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)#每个头的size
        self.all_head_size = self.num_attention_heads * self.attention_head_size#一共的size，隐层数目

        #自注意力的Q，K，V是三个linear变换得到，总参数是3*h^2
        self.query = nn.Linear(config.hidden_size, self.all_head_size)#隐层和head的大小是一样的
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        #原维度 (b, h, head_num, head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) #变换后 (b, head_num, h, head_size)，相当于分成了多头

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        #先得到Q，K，V
        mixed_query_layer = self.query(hidden_states)# (b, seq_len, h)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)# (b, seq_len, h)
            mixed_value_layer = self.value(encoder_hidden_states)# (b, seq_len, h)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        #再转换维度以多头，便于点乘
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        #计算Q和K的相似度
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#除K的维度放缩注意力分数
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # 掩码这里使用了加号，对于不掩码的位置来说，掩码值为0
            # 对于掩码的位置来说，掩码值为-10000。使用softmax层之后，可以让-10000处的值为0。
            attention_scores = attention_scores + attention_mask

        #将注意力分数分配成概率矩阵
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:#维度(b, num_head, seq_len, seq_len)，要么为0，要么为1
            attention_probs = attention_probs * head_mask

        #分配注意力概率给值V
        context_layer = torch.matmul(attention_probs, value_layer)
        #变化维度
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (b, seq_len, head_num, head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # (b, seq_len, h)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    '''这个是自注意力之后马上接的shortcut和layernorm
    '''
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states) #一层linear
        hidden_states = self.dropout(hidden_states) #dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor) #残差连一下
        return hidden_states


class BertAttention(nn.Module):
    '''多头注意力整和到一起，包括注意力后的add && norm部分
    '''
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        #修剪头
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )#自注意力
        attention_output = self.output(self_outputs[0], hidden_states)#add && norm
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    '''中间线性层，FeedForward和激活函数。就是多加了一层的抽象
    '''
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            #与Transformer不同，BERT用GELU
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states) #linear是FFN，投到高维度增强表达能力
        hidden_states = self.intermediate_act_fn(hidden_states)#然后激活
        return hidden_states


class BertOutput(nn.Module):
    '''输出层，在FFN后面加了shortcut和layer norm
    '''
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)#第二次linear
        hidden_states = self.dropout(hidden_states)#dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor)#残差连一下
        return hidden_states


class BertLayer(nn.Module):
    '''一层encoder，即一个Transformer（的encoder部分）
    '''
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        #注意力层（这个已经有了LN和残差）
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:#如果加了cross层
            #就计算cross的计算结果
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        #FFN
        intermediate_output = self.intermediate(attention_output)
        #LN和残差输出层
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    '''12层Encoder，即12个Transformer
    '''
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):#每一层堆叠
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            #注意力层
            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]
            #输出层
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        #加上最后一层
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        #所有层的结果
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    '''开始做两大任务，首先是NPS任务
    Pooler是把隐藏层(hidden state)中对应#CLS#的token的特征向量提取出来。
    对于分类问题来说，最后使用第一个字符[CLS]的表示来进行分类。[CLS]需要在微调阶段继续进行训练。
    非分类问题可忽略。
     '''
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]#第一个特征，即【CLS】的特征
        pooled_output = self.dense(first_token_tensor)#做 next sentence classification任务
        pooled_output = self.activation(pooled_output)#tanh激活，二分类
        return pooled_output
    

class BertPredictionHeadTransform(nn.Module):
    '''封装了一个顺序为：线性变换，激活，Layer Normal 的变换流程，为后续的 Masked language modeling 任务做准备。
    '''
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    '''LM任务。把上面的transformer输出[batch_size, seq_len, embed_dim]映射为[batch_size, seq_len, vocab_size]
      也就是把最后一个维度映射成字典中字的数量, 获取MaskedLM的预测结果。
      BERT是不知道谁被mask的，所以encoder的时候会保持所有输入token的上下文。
      注意这里其实也可以直接矩阵成embedding矩阵的转置, 但一般情况下我们要随机初始化新的一层参数
    '''
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)#刚刚封装的流程

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        #就是这里，从hidden_size到vocab_size的映射，就可得到mask部分应该是什么
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))#变一下维度

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias #最后加个偏置

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    '''prediction_scores 用于 mask language modeling 任务
    '''
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    '''seq_relationship_scre 用于 next sentence classfifcation 任务
    '''
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)#句子的关系

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    """
    BERT的训练中通过隐藏层输出Masked LM的预测和Next Sentence的预测
    """
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score #输出两个分数


class BertPreTrainedModel(PreTrainedModel):
    """ 初始化模型参数。An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig #得到配置信息
    # pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    # load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ 初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            #linear是正太分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()#LN的alpha是1，beta全0
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()#bias是0


class BertModel(BertPreTrainedModel):
    """这个是完整的BERT模型，注释一大堆，实际上就是把前面的部分组装起来
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        #实例话三个模块
        self.embeddings = BertEmbeddings(config)#把3个嵌入给处理好
        self.encoder = BertEncoder(config)#N个Transformer
        self.pooler = BertPooler(config)#得到【CLS】

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.
            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import BertModel, BertTokenizer
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        """
        #得到嵌入
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        #处理mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        
        # 这里的2D变3D多了一个seq_length的维度是在每个词里面再分一个维度，表示当前词能看到哪些词。
        # 如2个句子 [[1,2,3,4,0,0,0,0],[7,8,9,10,11,0,0,0]]，除了原先标记pad的[1,1,1,1,0,0,0,0]
        # 现在会多在句子id的1号词扩展[1,1,1,1,0,0,0,0]，表示这个词能与前4个算Attention
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        #embedding层
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )#Transformer的输出
        sequence_output = encoder_outputs[0]#最后的输出结果
        pooled_output = self.pooler(sequence_output)#【CLS】向量

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertForPreTraining(nn.Module):
    '''最终的预训练类。
    loss来自两部分，一部分是掩码模型的平均值，一部分是next sentence prediction的平均值。
    掩码部分真实标签值只有掩码单元有意义，其它部分均为-1，在计算loss的时候会忽略掉这部分。
    '''
    def __init__(self, config,model):
        super().__init__(config)

        self.bert = model#实例化BERT
        self.cls = BertPreTrainingHeads(config)#两个任务类

        # self.init_weights()#初始化权重

    def get_output_embeddings(self):
        return self.cls.predictions.decoder #cls的预测结果
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        # next_sentence_label=None,
    ):
        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`, defaults to :obj:`None`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False
            continuation before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import BertTokenizer, BertForPreTraining
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]
        """
        #拿到BERT的输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        #做两个任务
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        #计算loss
        # if masked_lm_labels is not None and next_sentence_label is not None:
        #     loss_fct = CrossEntropyLoss() #MLM用交叉熵，实际上会只算mask部分的loss，所以和DAE很像。
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        #     #NSP用cls的预测也是跟label算一次交叉熵
        #     next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
        #     total_loss = masked_lm_loss + next_sentence_loss
        #     outputs = (total_loss,) + outputs

        loss_fct = CrossEntropyLoss() #MLM用交叉熵，实际上会只算mask部分的loss，所以和DAE很像。
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        outputs = (masked_lm_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)

