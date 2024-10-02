import torch
import torch.nn as nn


class TemporalConvolutions(nn.Module):
    def __init__(self, n_layers=6, f_size=48, kernel_size=16, n_classes=4, temporal_length=120) -> None:
        super().__init__()

        self.initial_conv = nn.Sequential(nn.Conv1d(1, f_size, kernel_size=kernel_size, padding="same"), nn.ReLU())

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.Sequential(
                nn.Conv1d(f_size, f_size, kernel_size=kernel_size, padding="same"),
                nn.ReLU(),
            )
            self.layers.append(layer)

        self.final_layer = nn.Linear(f_size * temporal_length, n_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        x = self.final_layer(x)
        return x


class TemporalFullyConnected(nn.Module):
    def __init__(
        self,
        n_layers=6,
        f_size=48,
        n_classes=4,
        input_size=1,
        temporal_length=120,
        with_residual=True,
        dropout=0.5,
    ) -> None:
        super().__init__()

        self.initial_layer = nn.Sequential(
            nn.Linear(temporal_length * input_size, f_size),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList()
        self.with_residual = with_residual
        for i in range(n_layers):
            layer = nn.Sequential(
                nn.Linear(f_size, f_size),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            self.layers.append(layer)

        self.final_layer = nn.Linear(f_size, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = self.initial_layer(x)
        for layer in self.layers:
            if self.with_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        x = self.final_layer(x)
        return x


class TemporalRNN(nn.Module):
    def __init__(
        self,
        input_size=1,
        n_layers=2,
        hidden_size=48,
        n_classes=4,
        use_gru=False,
        bidirectional=False,
        **kwargs,
    ) -> None:
        super().__init__()
        if use_gru:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=bidirectional,
                **kwargs,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=bidirectional,
                **kwargs,
            )

        self.final_layer = nn.Linear(hidden_size if not bidirectional else hidden_size * 2, n_classes)

    def forward(self, x):
        x, hiddens = self.rnn(x)
        x = torch.max(x, 1)[0]
        x = self.final_layer(x)
        return x


class MixedFCTemporalModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        n_rnn_layers=2,
        n_fc_layers=2,
        temporal_length=120,
        bidirectional=True,
        rnn_hidden_size=32,
        fc_hidden_size=64,
        n_classes=4,
        dropout=0.25,
        pooling=None,
        use_gru=True,
        **kwargs,
    ) -> None:
        super().__init__()

        if use_gru:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=rnn_hidden_size,
                num_layers=n_rnn_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
                **kwargs,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=rnn_hidden_size,
                num_layers=n_rnn_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
                **kwargs,
            )
        fcs = []
        input_size = rnn_hidden_size * 2 ** (bidirectional)
        self.pooling = pooling
        if not pooling:
            input_size *= temporal_length
        for i in range(n_fc_layers):
            fc = nn.Sequential(
                nn.Linear(
                    input_size,
                    fc_hidden_size,
                ),
                nn.ReLU(),
                nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            )
            fcs.append(fc)
            input_size = fc_hidden_size
        self.fc = nn.Sequential(*fcs)

        self.final_layer = nn.Linear(fc_hidden_size, n_classes)

    def forward(self, x):
        x, hiddens = self.rnn(x)
        if self.pooling == "max":
            x = torch.max(x, 1)[0]
        elif self.pooling == "mean":
            x = torch.mean(x, 1)
        elif self.pooling == "last":
            x = x[:, -1]
        else:
            x = x.flatten(1)
        x = torch.relu(x)
        x = self.fc(x)
        x = self.final_layer(x)
        return x
