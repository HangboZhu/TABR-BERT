import torch
import torch.nn as nn


class TcrPmhcClassifier(nn.Module):
    def __init__(self, d_model: int = 1280, tcr_maxlen: int = 30, pmhc_maxlen: int = 54):
        super().__init__()
        self.d_model = d_model
        self.tcr_maxlen = tcr_maxlen
        self.pmhc_maxlen = pmhc_maxlen

        self.pmhc_linear = nn.Sequential(nn.Linear(d_model, 1))
        self.tcr_linear = nn.Sequential(nn.Linear(d_model, 1))
        self._pmhc_linear = nn.Sequential(nn.Linear(pmhc_maxlen, 1))
        self._tcr_linear = nn.Sequential(nn.Linear(tcr_maxlen, 1))

        self.dense = nn.Sequential(
            nn.Linear(pmhc_maxlen + tcr_maxlen + d_model * 2, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.tcr_maxlen + self.pmhc_maxlen, self.d_model)
        tcr = x[:, : self.tcr_maxlen, :]
        pmhc = x[:, self.tcr_maxlen :, :]

        tcr_x = self.tcr_linear(tcr).reshape(-1, self.tcr_maxlen)
        pmhc_x = self.pmhc_linear(pmhc).reshape(-1, self.pmhc_maxlen)
        _tcr_x = self._tcr_linear(tcr.transpose(1, 2)).reshape(-1, self.d_model)
        _pmhc_x = self._pmhc_linear(pmhc.transpose(1, 2)).reshape(-1, self.d_model)

        out = self.dense(torch.cat([tcr_x, _tcr_x, pmhc_x, _pmhc_x], dim=-1))
        return out.view(-1)
