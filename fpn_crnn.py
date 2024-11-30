import torch 
import torch.nn as nn
import warnings
from models.CNN import CNN
#test
#rom RNN import BidirectionalGRU
#runs
from .RNN import BidirectionalGRU

import torch.nn as nn
import torch
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res

class FPN_CNN_block(nn.Module):
    def __init__(
        self,
        cnn_channel=128,
        dropout = 0.5,
        kernel_size = (2,1),
        ):
        super(FPN_CNN_block, self).__init__()
        
        self.cnn_channel = cnn_channel
        self.dropout = dropout
        self.kernel_size = kernel_size
        
        self.fpn_conv_block = nn.Sequential(
                nn.Conv2d(self.cnn_channel, self.cnn_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.cnn_channel, eps=0.001, momentum=0.99),
                GLU(self.cnn_channel),
                nn.Dropout(self.dropout),
                nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
                        
    def forward(self,x):
        
        x = self.fpn_conv_block(x)
        
        return x
    
class FPN_RNN_block(nn.Module):
    def __init__(
        self,
        dropout_recurrent=0,
        n_RNN_cell=128,
        n_layers_RNN=2,
        rnn_type="BGRU",
        dropout = 0.5
    ):
        super(FPN_RNN_block, self).__init__()
        
        if rnn_type == "BGRU":
            nb_in=128
            
            self.fpn_rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
            self.drop_layer = nn.Dropout(dropout)
    
    def forward(self, x):
        
        x = self.fpn_rnn(x)
        x = self.drop_layer(x)
        bs,frame,feature = x.size()
        x = x.view(bs, 1, frame, feature)
        return x
        
            
class CRNN(nn.Module):
    def __init__(
        self,
        n_in_channel=1,
        nclass=10,
        attention=True,
        activation="glu",
        dropout=0.5,
        train_cnn=True,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        cnn_integration=False,
        freeze_bn=False,
        use_embeddings=False,
        embedding_size=527,
        embedding_type="global",
        frame_emb_enc_dim=128,
        aggregation_type="global",
        **kwargs,
    ):
        """
            Initialization of CRNN model
        
        Args:
            n_in_channel: int, number of input channel
            n_class: int, number of classes
            attention: bool, adding attention layer or not
            activation: str, activation function
            dropout: float, dropout
            train_cnn: bool, training cnn layers
            rnn_type: str, rnn type
            n_RNN_cell: int, RNN nodes
            n_layer_RNN: int, number of RNN layers
            dropout_recurrent: float, recurrent layers dropout
            cnn_integration: bool, integration of cnn
            freeze_bn: 
            **kwargs: keywords arguments for CNN.
        """
        super(CRNN, self).__init__()

        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.freeze_bn = freeze_bn
        self.use_embeddings = use_embeddings
        self.embedding_type = embedding_type
        self.aggregation_type = aggregation_type

        n_in_cnn = n_in_channel
        #hyper parameter
        fpn_cnn_channel = kwargs['fpn_cnn_channel']
        fpn_cnn_dropout = kwargs['fpn_cnn_dropout']
        fpn_cnn_kernel_size = kwargs['fpn_cnn_kernel_size']
        
        fpn_rnn_dropout_recurrent= kwargs['fpn_rnn_dropout_recurrent']
        fpn_rnn_n_RNN_cell = kwargs['fpn_rnn_n_RNN_cell']
        fpn_n_layers_RNN = kwargs['fpn_n_layers_RNN']
        fpn_rnn_type = kwargs['fpn_rnn_type']
        fpn_rnn_dropout = kwargs['fpn_rnn_dropout']
        
        scale_factor = kwargs['scale_factor']
        scale_factor = tuple(scale_factor)
        fpn_block_num = kwargs['fpn_block_num']
        self.upsample_type = kwargs['upsample_type']
        fpn_point_wise_chan = kwargs['fpn_point_wise_chan']
        fpn_point_wise_dropout = kwargs['fpn_point_wise_dropout']
        
        if cnn_integration:
            n_in_cnn = 1

        self.cnn = CNN(
            n_in_channel=n_in_cnn,activation=activation, conv_dropout=dropout, **kwargs
        )
        
        self.train_cnn = train_cnn
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
                
        self.fpn_conv_block = nn.ModuleList()
        self.fpn_rnn_block = nn.ModuleList()
        
        self.upsample = nn.ModuleList()
        self.pointwise_conv = nn.ModuleList()

        def append_fpn_block(block_num):
            for i in range(block_num):
                self.fpn_conv_block.append(FPN_CNN_block(
                    cnn_channel=fpn_cnn_channel,
                    dropout=fpn_cnn_dropout,
                    kernel_size=fpn_cnn_kernel_size
                    ))
                self.fpn_rnn_block.append(FPN_RNN_block(
                    dropout_recurrent=fpn_rnn_dropout_recurrent,
                    n_RNN_cell=fpn_rnn_n_RNN_cell,
                    n_layers_RNN=fpn_n_layers_RNN,
                    rnn_type=fpn_rnn_type,
                    dropout=fpn_rnn_dropout
                ))
                self.upsample.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False))

                if self.upsample_type == 'cat':
                    if i == 0:
                        self.pointwise_conv.append(nn.Sequential(
                            nn.Conv2d(2, fpn_point_wise_chan, kernel_size=1, stride=1, padding='same'),
                            nn.BatchNorm2d(fpn_point_wise_chan, eps=0.001, momentum=0.99),
                            nn.ReLU(),
                            nn.Dropout(fpn_point_wise_dropout)
                            ))      
                    elif i > 0 and i < block_num-1:
                        self.pointwise_conv.append(nn.Sequential(
                            nn.Conv2d(fpn_point_wise_chan+1, fpn_point_wise_chan, kernel_size=1, stride=1, padding='same'),
                            nn.BatchNorm2d(fpn_point_wise_chan, eps=0.001, momentum=0.99),
                            nn.ReLU(),
                            nn.Dropout(fpn_point_wise_dropout)
                            ))
                    else:
                        self.pointwise_conv.append(nn.Sequential(
                            nn.Conv2d(fpn_point_wise_chan+1, 1, kernel_size=1, stride=1, padding='same'),
                            nn.BatchNorm2d(1, eps=0.001, momentum=0.99),
                            nn.ReLU(),
                            nn.Dropout(fpn_point_wise_dropout)
                            ))
                else :
                    if i == 0:
                        self.pointwise_conv.append(nn.Sequential(
                            nn.Conv2d(1, fpn_point_wise_chan, kernel_size=1, stride=1, padding='same'),
                            nn.BatchNorm2d(fpn_point_wise_chan, eps=0.001, momentum=0.99),
                            nn.ReLU(),
                            nn.Dropout(fpn_point_wise_dropout)
                            ))      
                    elif i > 0 and i < block_num-1:
                        self.pointwise_conv.append(nn.Sequential(
                            nn.Conv2d(fpn_point_wise_chan, fpn_point_wise_chan, kernel_size=1, stride=1, padding='same'),
                            nn.BatchNorm2d(fpn_point_wise_chan, eps=0.001, momentum=0.99),
                            nn.ReLU(),
                            nn.Dropout(fpn_point_wise_dropout)
                            ))
                    else:
                        self.pointwise_conv.append(nn.Sequential(
                            nn.Conv2d(fpn_point_wise_chan, 1, kernel_size=1, stride=1, padding='same'),
                            nn.BatchNorm2d(1, eps=0.001, momentum=0.99),
                            nn.ReLU(),
                            nn.Dropout(fpn_point_wise_dropout)
                            ))
                          

        append_fpn_block(block_num=fpn_block_num)

        if rnn_type == "BGRU":
            nb_in = 128
            if self.cnn_integration:
                # self.fc = nn.Linear(nb_in * n_in_channel, nb_in)
                nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.sigmoid = nn.Sigmoid()
        
        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, nclass)
            self.softmax = nn.Softmax(dim=-1)
            
        if self.use_embeddings:
            if self.aggregation_type == "frame":
                self.frame_embs_encoder = nn.GRU(batch_first=True, input_size=embedding_size,
                                                      hidden_size=128,
                                                      bidirectional=True)
                self.shrink_emb = torch.nn.Sequential(torch.nn.Linear(2 * frame_emb_enc_dim, nb_in),
                                                      torch.nn.LayerNorm(nb_in))
                self.cat_tf = torch.nn.Linear(2*nb_in, nb_in)
            elif self.aggregation_type == "global":
                self.shrink_emb = torch.nn.Sequential(torch.nn.Linear(embedding_size, nb_in),
                                                      torch.nn.LayerNorm(nb_in))
                self.cat_tf = torch.nn.Linear(2*nb_in, nb_in)
            elif self.aggregation_type == "interpolate":
                self.cat_tf = torch.nn.Linear(nb_in+embedding_size, nb_in)
            elif self.aggregation_type == "pool1d":
                self.cat_tf = torch.nn.Linear(nb_in+embedding_size, nb_in)
            else:
                self.cat_tf = torch.nn.Linear(2*nb_in, nb_in)      
    
    def upsampling(self, x , y , num):
        
        x = self.upsample[num](x)
       
        if self.upsample_type == 'cat':
            x = torch.cat((x,y),dim=1)
        else :
            x = x + y
            
        x = self.pointwise_conv[num](x)
        return x 
                    
    
   
   
   
    def embedding(self, x, embeddings):
        if self.use_embeddings:
            if self.aggregation_type == "global":
                x = self.cat_tf(torch.cat((x, self.shrink_emb(embeddings).unsqueeze(1).repeat(1, x.shape[1], 1)), -1))
            elif self.aggregation_type == "frame":
                # there can be some mismatch between seq length of cnn of crnn and the pretrained embeddings, we use an rnn
                # as an encoder and we use the last state
                last, _ = self.frame_embs_encoder(embeddings.transpose(1, 2))
                embeddings = last[:, -1]
                x = self.cat_tf(torch.cat((x, self.shrink_emb(embeddings).unsqueeze(1).repeat(1, x.shape[1], 1)), -1))
            elif self.aggregation_type == "interpolate":
                output_shape = (embeddings.shape[1], x.shape[1])
                reshape_emb = torch.nn.functional.interpolate(embeddings.unsqueeze(1), size=output_shape, mode='nearest-exact').squeeze(1).transpose(1, 2)
                x = self.cat_tf(torch.cat((x, reshape_emb), -1))
            elif self.aggregation_type == "pool1d":
                reshape_emb = torch.nn.functional.adaptive_avg_pool1d(embeddings, x.shape[1]).transpose(1, 2)
                x = self.cat_tf(torch.cat((x, reshape_emb), -1))
            else:
                return x
            
        return x
   
  
   

   
   
   
   
   
    
    
    
    def size_rewind(self, x):
        bs, chan, frames, freq = x.size()
        
        if freq != 1:
            warnings.warn(
                f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq"
            )
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)

        return x
    
    def fpn_block(self, x, num):
        cnn_block = self.fpn_conv_block[num](x)
        rnn_block = self.size_rewind(cnn_block)
        # rnn_block = self.embedding(rnn_block, embeddings=None)
        rnn_block = self.fpn_rnn_block[num](rnn_block)
        return cnn_block,rnn_block
        
    def forward(self, x, pad_mask=None, embeddings=None):
        # 초기 x 크기 출력
          # 예: torch.Size([24, 1, 628, 128])

        # 전치 및 리쉐이프
        x = x.transpose(1, 2)
          # 예: torch.Size([24, 628, 1, 128])
        bs, n_frames, n_channels, n_freq = x.size()
        x = x.view(bs, n_channels, n_frames, n_freq)
        # 예: torch.Size([24, 1, 628, 128])

        # cnn_integration 여부 확인
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # CNN 레이어 통과 후 크기 출력
        cnn_block0 = self.cnn(x)
       

        # RNN 처리
        rnn_block0 = self.size_rewind(cnn_block0)
        

        rnn_block0 = self.embedding(rnn_block0, embeddings)
        

        rnn_block0 = self.rnn(rnn_block0)
        

        rnn_block0 = self.dropout(rnn_block0)
        

        # RNN 블록 리쉐이프
        bs, frame, feature = rnn_block0.size()
        rnn_block0 = rnn_block0.view(bs, 1, frame, feature)
        

        # FPN 블록 통과 후 크기 출력
        cnn_block1, rnn_block1 = self.fpn_block(cnn_block0, num=0)
        

        cnn_block2, rnn_block2 = self.fpn_block(cnn_block1, num=1)
        

        # Upsampling
        x = self.upsampling(rnn_block2, rnn_block1, num=0)
        

        x = self.upsampling(x, rnn_block0, num=1)
        

        x = x.squeeze(1)
         # 예: torch.Size([24, 157, 256])
    
    
       
        
        
        
        
        
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            if not pad_mask is None:
                sof = sof.masked_fill(pad_mask.transpose(1, 2), -1e30)  # mask attention
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)
        #return strong.transpose(1, 2), weak
        return strong, weak



if __name__ == '__main__':
    CRNN(64, 10, kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
         pooling=[(1, 4), (1, 4), (1, 4)])