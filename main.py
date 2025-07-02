import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# モデルの定義
class Encoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(28 * 28 + 16, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, image, label):
        flattened_image = image.view(image.size(0), -1)
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_input = torch.cat([flattened_image, label_embedding], dim=1)
        hidden_activation = F.relu(self.fc_hidden(concatenated_input))
        return self.fc_mu(hidden_activation), self.fc_logvar(hidden_activation)

class Decoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(latent_dim + 16, 128)
        self.fc_out = nn.Linear(128, 28 * 28)

    def forward(self, latent_vector, label):
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_latent = torch.cat([latent_vector, label_embedding], dim=1)
        hidden_activation = F.relu(self.fc_hidden(concatenated_latent))
        output_linear = self.fc_out(hidden_activation)
        return torch.sigmoid(output_linear).view(-1, 1, 28, 28)

class CVAE(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのロード
model = CVAE(latent_dim=3, num_classes=10).to(device)
model.load_state_dict(torch.load("cvae.pth", map_location=device))
model.eval()

# Streamlit UI
st.title("Conditional Variational Autoencoder (CVAE) Generator")
st.write("### 数字を選択して、対応する画像を生成")

# サイドバーで生成する枚数を指定
num_samples = st.sidebar.slider("生成する枚数", min_value=1, max_value=10, value=1)
digit = st.sidebar.selectbox("生成する数字 (0-9)", list(range(10)), index=0)

generate_button = st.sidebar.button("画像を生成")

if generate_button:
    z_random_vector = torch.randn(num_samples, 3).to(device)
    label = torch.full((num_samples,), digit, dtype=torch.long, device=device)
    
    with torch.no_grad():
        x_gen = model.decoder(z_random_vector, label)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    if num_samples == 1:
        axes = [axes]
    for i in range(num_samples):
        axes[i].imshow(x_gen[i].squeeze().cpu().numpy(), cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)
