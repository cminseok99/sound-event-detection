import torch.nn as nn
import torch


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


class CNN(nn.Module):

    def __init__(self,n_in_channel, activation="Relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],fpn_cnn_channel=128,fpn_cnn_dropout=0.5, fpn_cnn_kernel_size=3,
                 fpn_rnn_dropout_recurrent=0.5, fpn_rnn_n_RNN_cell=128, fpn_n_layers_RNN=2, fpn_rnn_type="BIGU",
                 fpn_rnn_dropout=0.5, scale_factor=[2, 2], fpn_block_num=2, upsample_type="cat", 
                 fpn_point_wise_chan=1, fpn_point_wise_dropout=0.5
                 ):
        super(CNN, self).__init__()
        self.fpn_cnn_channel = fpn_cnn_channel
        self.fpn_cnn_dropout = fpn_cnn_dropout
        self.fpn_cnn_kernel_size = fpn_cnn_kernel_size
        self.fpn_rnn_dropout_recurrent = fpn_rnn_dropout_recurrent
        self.fpn_rnn_n_RNN_cell = fpn_rnn_n_RNN_cell
        self.fpn_n_layers_RNN = fpn_n_layers_RNN
        self.fpn_rnn_type = fpn_rnn_type
        self.fpn_rnn_dropout = fpn_rnn_dropout
        self.scale_factor = scale_factor
        self.fpn_block_num = fpn_block_num
        self.upsample_type = upsample_type
        self.fpn_point_wise_chan = fpn_point_wise_chan
        self.fpn_point_wise_dropout = fpn_point_wise_dropout
        self.nb_filters = nb_filters # 7개

        self.conv0 = nn.Conv2d(1,16,3,1,1)
        self.batchnorm0 = nn.BatchNorm2d(16, eps=0.001, momentum=0.99)
        self.glu0 = GLU(16)
        self.dropout0 = nn.Dropout(0.5)
        self.pooling0 = nn.AvgPool2d([2,2])

        self.conv1 = nn.Conv2d(16,32,3,1,1)
        self.batchnorm1 = nn.BatchNorm2d(32, eps=0.001, momentum=0.99)
        self.glu1 = GLU(32)
        self.dropout1 = nn.Dropout(0.5)
        self.pooling1 = nn.AvgPool2d([2,2])

        self.conv2 = nn.Conv2d(32,64,3,1,1)
        self.batchnorm2 = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)
        self.glu2 = GLU(64)
        self.dropout2 = nn.Dropout(0.5)
        self.pooling2 = nn.AvgPool2d([1,2])

        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.batchnorm3 = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)
        self.glu3 = GLU(128)
        self.dropout3 = nn.Dropout(0.5)
        self.pooling3 = nn.AvgPool2d([1,2])

        self.conv4 = nn.Conv2d(128,128,3,1,1)
        self.batchnorm4 = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)
        self.glu4 = GLU(128)
        self.dropout4 = nn.Dropout(0.5)
        self.pooling4 = nn.AvgPool2d([1,2])
        

        self.conv5 = nn.Conv2d(128,128,3,1,1)
        self.batchnorm5 = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)
        self.glu5 = GLU(128)
        self.dropout5 = nn.Dropout(0.5)
        self.pooling5 = nn.AvgPool2d([1,2])

        self.conv6 = nn.Conv2d(128,128,3,1,1)
        self.batchnorm6 = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)
        self.glu6 = GLU(128)
        self.dropout6 = nn.Dropout(0.5)
        self.pooling6 = nn.AvgPool2d([1,2])
        
        self.linear = nn.Linear(128, 128)  # Define the linear layer with appropriate input/output dimensions
        self.sigmoid = nn.Sigmoid()

        
        #   "nb_filters": [16,  32,  64,  128,  128, 128, 128],
        # "nb_filters": [16,  32,  64,  84,  104, 114, 128],

        # def conv(i, batchNormalization=False, dropout=None, activ="relu"):
        #     nIn = n_in_channel if i == 0 else nb_filters[i - 1]
        #     nOut = nb_filters[i]
        #     cnn.add_module('conv{0}'.format(i),
        #                    nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
        #     if batchNormalization:
        #         cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
        #     if activ.lower() == "leakyrelu":
        #         cnn.add_module('relu{0}'.format(i),
        #                        nn.LeakyReLU(0.2))
        #     elif activ.lower() == "relu":
        #         cnn.add_module('relu{0}'.format(i), nn.ReLU())
        #     elif activ.lower() == "glu":
        #         cnn.add_module('glu{0}'.format(i), GLU(nOut))
        #     elif activ.lower() == "cg":
        #         cnn.add_module('cg{0}'.format(i), ContextGating(nOut))
        #     if dropout is not None:
        #         cnn.add_module('dropout{0}'.format(i),
        #                        nn.Dropout(dropout))
        #     cnn.add_module('pooling{0}'.format(i), nn.AvgPool2d(pooling[i]))
            
        
        # batch_norm = True

      #128x862x64
        # for i in range(len(nb_filters)):
        #     conv(i,batch_norm, conv_dropout, activ=activation)
        
        # self.cnn = cnn

    # def load_state_dict(self, state_dict, strict=True):
    #     self.cnn.load_state_dict(state_dict)

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     return self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    # def save(self, filename):
    #     torch.save(self.cnn.state_dict(), filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
      
        x =  self.conv0(x)
        x =  self.batchnorm0(x)
        x =  self.glu0(x)
        x =  self.dropout0(x) 
        x =  self.pooling0(x)

        x =  self.conv1(x)
        x =  self.batchnorm1(x)
        x =  self.glu1(x)
        x =  self.dropout1(x) 
        x =  self.pooling1(x)

        x =  self.conv2(x)
        x =  self.batchnorm2(x)
        x =  self.glu2(x)
        x =  self.dropout2(x) 
        x =  self.pooling2(x)

        x =  self.conv3(x)
        x =  self.batchnorm3(x)
        x =  self.glu3(x)
        x =  self.dropout3(x) 
        x =  self.pooling3(x)

        x =  self.conv4(x)
        x =  self.batchnorm4(x)
        x =  self.glu4(x)
        x =  self.dropout4(x) 
        x =  self.pooling4(x)

        x =  self.conv5(x)
        x =  self.batchnorm5(x)
        x =  self.glu5(x)
        x =  self.dropout5(x) 
        x =  self.pooling5(x)

        x =  self.conv6(x)
        x =  self.batchnorm6(x)
        x =  self.glu6(x)
        x =  self.dropout6(x) 
        x =  self.pooling6(x)
        
        return x[:,:,:-1,:]


