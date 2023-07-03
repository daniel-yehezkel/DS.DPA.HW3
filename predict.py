from dataset import HW3Dataset
import torch
import os
import pandas as pd
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(data.num_features + 1, hidden_channels, heads=7)
        self.conv2 = GATConv(hidden_channels * 7, hidden_channels, heads=6)
        self.conv3 = GATConv(hidden_channels * 6, data.num_classes, heads=6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv3(x, edge_index)
        return x
    
    
if __name__ == "__main__":
    print("Strating..")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    data = HW3Dataset(root='data/hw3/')

    min_value = 1971
    max_value = 2017
    diff = 46

    model = torch.load(os.path.join("weights", "model"))
    model = model.to(device)
    model.eval()

    out = model(
        torch.concatenate((data[0].x, (data[0].node_year - min_value) / diff), axis=1).to(device), 
        data[0].edge_index.to(device)
    )
    pred = out.argmax(dim=1).detach().cpu()
    df = pd.DataFrame([range(len(pred)), [int(x) for x in pred]]).T
    df.columns = ["idx", "prediction"]
    df.to_csv("prediction.csv", index=False)
    print("Done")
