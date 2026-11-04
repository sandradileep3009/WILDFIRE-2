import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, h_prev, c_prev):

        combined = torch.cat([x, h_prev], dim=1)

        gates = self.conv(combined)

        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):

        height, width = spatial_size

        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

        return h, c


class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dims, kernel_size=3):
        super().__init__()

        self.layers = nn.ModuleList()

        for i, hidden_dim in enumerate(hidden_dims):

            layer_input_dim = input_dim if i == 0 else hidden_dims[i - 1]

            self.layers.append(
                ConvLSTMCell(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                )
            )

    def forward(self, x):

        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        height = x.shape[-2]
        width = x.shape[-1]

        layer_input = x

        for layer in self.layers:

            h, c = layer.init_hidden(
                batch_size=batch_size,
                spatial_size=(height, width),
                device=x.device,
            )

            outputs = []

            for t in range(sequence_length):
                h, c = layer(layer_input[:, t], h, c)
                outputs.append(h)

            layer_input = torch.stack(outputs, dim=1)

        return layer_input[:, -1]


class SpreadPredictionModel(nn.Module):
    """
    3 RGB frames + current fire mask -> future mask logits (or growth logits,
    depending on what target the trainer passes in).
    """

    def __init__(self, in_channels=3, hidden_dims=(32, 32)):
        super().__init__()

        self.convlstm = ConvLSTM(
            input_dim=in_channels,
            hidden_dims=list(hidden_dims),
            kernel_size=3,
        )

        feature_channels = hidden_dims[-1]

        # temporal features (feature_channels) + current mask (1 channel)
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_channels + 1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x, current_mask):
        

        if x.ndim != 5:
            raise ValueError(f"Expected input shape [B,T,C,H,W], got {tuple(x.shape)}")

        if current_mask.ndim != 4:
            raise ValueError(
                f"Expected current mask shape [B,1,H,W], got {tuple(current_mask.shape)}"
            )

        features = self.convlstm(x)

        if current_mask.shape[-2:] != features.shape[-2:]:
            current_mask = F.interpolate(
                current_mask, size=features.shape[-2:], mode="nearest"
            )

        conditioned = torch.cat([features, current_mask], dim=1)

        logits = self.decoder(conditioned)

        return logits

    @torch.no_grad()
    def predict_proba(self, x, current_mask):
        return torch.sigmoid(self.forward(x, current_mask))
