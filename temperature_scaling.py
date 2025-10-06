## temperature_scaling.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaling(nn.Module):
    def __init__(self, init_temp=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
    def forward(self, logits):
            # Move self.temperature to the same device as logits
        temperature = self.temperature.to(logits.device)
        return logits / temperature

    def calibrate(self, model, loader, device, mode="root_probability"):
        model.eval()
        logits_list, targets_list = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                if mode == "root_probability":
                    prob = out.squeeze().clamp(1e-6, 1-1e-6)
                    logits = torch.stack([torch.log(1-prob), torch.log(prob)], dim=1)
                    target = (batch.y.squeeze() > 0.5).long()
                elif mode == "distribution":
                    logits = out
                    target = batch.y.argmax(dim=1)
                else:
                    continue
                logits_list.append(logits.cpu())
                targets_list.append(target.cpu())
        logits = torch.cat(logits_list).to(device)
        targets = torch.cat(targets_list).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), targets)
            loss.backward()
            return loss
        optimizer.step(closure)
        return self.temperature.item()

    def apply(self, model, loader, device, mode="root_probability"):
        preds, targets = [], []
        model.eval(); self.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                if mode == "root_probability":
                    prob = out.squeeze().clamp(1e-6, 1-1e-6)
                    logits = torch.stack([torch.log(1-prob), torch.log(prob)], dim=1)
                    scaled_logits = self.forward(logits)
                    prob_calib = F.softmax(scaled_logits, dim=1)[:,1]
                    preds.append(prob_calib.cpu())
                    targets.append(batch.y.squeeze().cpu())
                elif mode == "distribution":
                    scaled_logits = self.forward(out)
                    prob_calib = F.softmax(scaled_logits, dim=1)
                    preds.append(prob_calib.cpu())
                    targets.append(batch.y.cpu())
        if preds:
            preds = torch.cat(preds)
            targets = torch.cat(targets)
        else:
            preds, targets = None, None
        return preds, targets
