import logging
import torch
import train_fns.lstm as lstm
from torch import nn
from torch.autograd import Variable

logger = logging.getLogger(__name__)


class DecoderLSTM(nn.Module):
    def __init__(self, cfg, input_size, hidden_size, output_size, gpu_id):
        self.cfg = cfg
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = lstm.LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.gpu_id = gpu_id

    def forward(self, inputs, hidden, state):
        output, (hidden, state) = self.lstm(inputs, (hidden, state))
        output = self.out(output)
        return output, hidden, state

    def initHidden(self):
        # h = Variable(torch.zeros(1, 1, self.hidden_size))
        # c = Variable(torch.zeros(1, 1, self.hidden_size))

        h = Variable(torch.randn(1, 1, self.hidden_size).float())
        c = Variable(torch.randn(1, 1, self.hidden_size).float())
        
        return h.cuda(self.gpu_id), c.cuda(self.gpu_id)
