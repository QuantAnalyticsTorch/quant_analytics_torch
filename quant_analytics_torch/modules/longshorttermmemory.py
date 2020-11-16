import torch

from quant_analytics_torch.modules.baseneuralnetworkmodule import BaseNeuralNetworkModule

class LongShortTermModule(BaseNeuralNetworkModule):
    def __init__(self, input_size=1, batch_size=10, hidden_layer_size=128, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.batch_size = batch_size

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        self.linear = torch.nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,batch_size,self.hidden_layer_size),
                            torch.zeros(1,batch_size,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out)
        return predictions