import torch
import torch.nn as nn
import torchvision
from functools import reduce
import torch.nn.functional as functional
from torch.autograd import Variable
criterion1 = nn.BCEWithLogitsLoss()
criterion2 = nn.L1Loss()

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.features = []
        self.features.append(nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

        ))
        self.features.append(nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),

        ))
        self.features.append(nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        ))
        self.features.append(nn.Sequential(
            nn.MaxPool2d(1, stride=1, ceil_mode=True),  # 1/4
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        ))
        self.features.append(nn.Sequential(
            nn.MaxPool2d(1, stride=1, ceil_mode=True),  # 1/4
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
        ))
        self.features = nn.ModuleList(self.features)
        vgg16 = torchvision.models.vgg16(pretrained=True)
        L_vgg16 = list(vgg16.features)
        L_self = reduce(lambda x,y: list(x)+list(y), self.features)
        for l1, l2 in zip(L_vgg16, L_self):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

    def forward(self, input):
        """
        input: 4D tensor [N, C, H, W]
        output: a list of 4D tensors
        """
        output = []
        for f in self.features:
            input = f(input)
            output.append(input)
        return output


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor.
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda()),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda()))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, lstm_input_dim, hidden_dim, kernel_size, num_levels,
                 batch_size = 1, bias=True, return_all_levels=False, light_field=True):
        """
        Initialize ConvLSTM.

        Parameters
        ----------
        input_size: [(int, int), (int, int), (int, int), (int, int), (int, int)]
            List of height and width of input tensors.
        input_dim: [int, int, int, int, int]
            List of number of channels of input tensors.
        lstm_input_dim: int
            Number of channels of input tensor of ConvLSTM. Keep the same at all levels.
        hidden_dim: int
            Number of channels of hidden state. Keep the same at all levels.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        num_levels: int
            Number of levels. 
        batch_size: int
            Size of batch.
        bias: bool
            Whether or not to add the bias.
        return_all_levels: bool
            Whether return the output tensor of all levels.
        light_field: bool
            Whether the input is light field focus stack.
        """

        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that `lstm_input_dim`, `kernel_size` and `hidden_dim` are lists having len == num_levels
        lstm_input_dim = self._extend_for_multilevel(lstm_input_dim, num_levels)
        hidden_dim = self._extend_for_multilevel(hidden_dim, num_levels)
        kernel_size = self._extend_for_multilevel(kernel_size, num_levels)

        if not len(kernel_size) == len(hidden_dim) == num_levels:
            raise ValueError('Inconsistent list length.')

        self.input_size = input_size
        self.input_dim = input_dim
        self.lstm_input_dim = lstm_input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_levels = num_levels
        self.batch_size = batch_size
        self.bias = bias
        self.return_all_levels = return_all_levels
        self.light_field = light_field

        reduce_list = []
        forward_cell = [] # ConvLSTM cell
        attention_cell = [] #attentive module
        pred_list = []

        for i in range(0, self.num_levels):
            reduce_list.append(nn.Sequential(
                nn.Conv2d(self.input_dim[i], self.lstm_input_dim[i], 1),
                nn.Tanh(),))

            attention_cell.append(nn.Sequential(
                nn.Conv2d(in_channels=2*self.lstm_input_dim[i],
                         out_channels=2,
                         kernel_size=3,
                         padding=1,
                         bias=self.bias),
                nn.Softmax(),)
                )

            forward_cell.append(ConvLSTMCell(input_size=self.input_size[i],
                                          input_dim=self.lstm_input_dim[i],
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

            pred_list.append(nn.Conv2d(self.hidden_dim[i], 1, 1))

        self.feature = Feature()
        self.reduce_list = nn.ModuleList(reduce_list)
        self.attention_cell = nn.ModuleList(attention_cell)
        self.forward_cell = nn.ModuleList(forward_cell)
        self.pred_list = nn.ModuleList(pred_list)

    def forward(self, img):
        # If the input is not light field focus stack, repeat the img for num_step times. 
        if not self.light_field:
            num_step = 3
            feats = self.feature(img)
            feats.reverse()
            input_tensor = [feats] * num_step
        else:
            num_step = img.size(1)
            input_tensor = []
            for i in range(num_step):
                feats = self.feature(img[:, i])
                feats.reverse()
                input_tensor.append(feats)

        hidden_state = self._init_hidden(batch_size=self.batch_size)
        pred = []
        
        for idx in range(self.num_levels):
            h, c = hidden_state[idx]
            hidden_list = []

            # At top level, initialize the output of previous level to zeros.
            if idx==0:
                pre_level_out = h
            # Upsample pre_level_out to same size as current level.
            else:
                pre_level_out = functional.interpolate(pre_level_out, size=self.input_size[idx], mode='bilinear')
                
            for t in range(num_step):
                cur_step_feature = self.reduce_list[idx](input_tensor[t][idx])

                # Compute the attention maps.
                att = self.attention_cell[idx](torch.cat((cur_step_feature, pre_level_out), dim=1))
                cur_step_input = cur_step_feature*att[:, 0:1] + pre_level_out*att[:, 1:]

                h, c = self.forward_cell[idx](input_tensor=cur_step_input, cur_state=[h, c])
                hidden_list.append(h)

            # Average the hidden state at all time steps.
            cur_level_out = torch.stack(hidden_list)
            cur_level_out = torch.mean(cur_level_out, dim=0)


            p = self.pred_list[idx](cur_level_out)
            pred.append(p)

            pre_level_out = cur_level_out

        if self.return_all_levels:
            return pred
        return pred[-1]

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_levels):
            init_states.append(self.forward_cell[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilevel(param, num_levels):
        if not isinstance(param, list):
            param = [param] * num_levels
        return param
