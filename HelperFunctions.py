import streamlit as st
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import io


class ConditionalAugmentation(nn.Module):
    def __init__(self, text_dim, projected_dim):
        super(ConditionalAugmentation, self).__init__()
        self.proj = nn.Linear(text_dim, projected_dim * 2)

    def forward(self, text_embedding):
        mu_logvar = self.proj(text_embedding)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Stage1Generator(nn.Module):
    def __init__(self, text_embedding_dim, noise_dim, img_size):
        super(Stage1Generator, self).__init__()
        self.fc1 = nn.Linear(768 + noise_dim, 128 * 8 * 8)
        self.reduced_embeddings = nn.Linear(text_embedding_dim, 128)
        self.bn1 = nn.BatchNorm1d(128 * 8 * 8)
        self.relu = nn.ReLU(inplace=True)
        self.upsample1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.upsample2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.upsample3 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        self.augment  = ConditionalAugmentation(768,768)
        self.img_size = img_size
        

    def forward(self, text_embedding, noise):
         
        x = self.augment(text_embedding)
        x = torch.cat((x, noise), dim=1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 128, 8, 8)
        x = self.relu(self.bn2(self.upsample1(x)))
        x = self.relu(self.bn3(self.upsample2(x)))
        x = self.tanh(self.upsample3(x))
        return x


stage1_generator = Stage1Generator(text_embedding_dim=768, noise_dim=100, img_size=64)



class Stage2Generator(nn.Module):
    def __init__(self, text_embedding_dim, img_size):
        super(Stage2Generator, self).__init__()
        self.fc1 = nn.Linear(text_embedding_dim + 3 * img_size * img_size, 128 * 16 * 16)
        self.bn1 = nn.BatchNorm1d(128 * 16 * 16)
        self.relu = nn.ReLU(inplace=True)
        self.upsample1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.upsample2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.upsample3 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        self.augment  = ConditionalAugmentation(768,768)
        self.img_size = img_size

    def forward(self, text_embedding, stage1_img):
        stage1_img_flat = stage1_img.view(stage1_img.size(0), -1)
        text_embedding = self.augment(text_embedding)
        x = torch.cat((text_embedding, stage1_img_flat), dim=1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 128, 16, 16)
        x = self.relu(self.bn2(self.upsample1(x)))
        x = self.relu(self.bn3(self.upsample2(x)))
        x = self.tanh(self.upsample3(x))
        return x


stage2_generator = Stage2Generator(text_embedding_dim=768, img_size=64)
# Set the model to evaluation mode
stage1_generator.eval()
stage2_generator.eval()
device = 'cpu'
stage1_generator.load_state_dict(torch.load('Weights/stage1Generator_weights.pth',map_location=device))
stage2_generator.load_state_dict(torch.load('Weights/stage2Generator_weights_UPDATED.pth',map_location=device))
print("Models loaded successfully")
    
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').eval()
    
print("bert loaded")


def Tokenize(sentence):
    encoded_input = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=64)
    with torch.no_grad():
            model_output = bert_model(**encoded_input)
    text_embedding = model_output.last_hidden_state.mean(dim=1).squeeze()

    return text_embedding.unsqueeze(0)
    


def generate_images(text_embeddings):
    noise = torch.randn(1, 100)   
    with torch.no_grad():
        Image_stage1 = stage1_generator(text_embeddings,noise)
        Image_stage2 = stage2_generator(text_embeddings,Image_stage1) 
    print(Image_stage2.shape)
    return Image_stage2.squeeze()

# def display_images(image, title="Generated Images"):
#     # Display a grid of images using matplotlib
#     # fig, axes = plt.subplots(4, 4, figsize=(10, 10))
#     # for i, ax in enumerate(axes.flatten()):
#     # ax.axis('off')
#     image = image.permute(1, 2, 0).to('cpu').detach().numpy()
#     plt.imshow(image)
#     # ax.imshow(image)
#     plt.show()
#     # ax.axis('off')
#     # plt.imshow


# st.title("Pokémon Image Generator")

# st.markdown('<div class="custom-label">Enter a sentence:</div>', unsafe_allow_html=True)
# input_text = st.text_input("", key='input', help='Type your sentence here', label_visibility='collapsed')
# # sentence = "A cheerful Bulbasaur ready for its next Pokémon adventure."
# # generate_images(Tokenize(sentence))
# # display_images(generate_images(Tokenize(input_text)))
# # print(Tokenize(sentence).shape)


# # Generate images
# st.write("Generating images...")
# # # # Replace with actual text embeddings input
# # # text_embeddings = torch.randn(16, 1024)  # Placeholder, use actual text embeddings

# if st.button("Generate Image"):
#     if input_text:
#         # generated_image = generate_image(input_text)
#         generated_image = generate_images(Tokenize(input_text))
#         img_bytes = io.BytesIO()
#         generated_image.save(img_bytes, format='PNG')
#         img_bytes.seek(0)
        
#         st.image(img_bytes, caption="Generated Image", use_column_width=True)
    
#     else:
#         st.error("Please enter a sentence.")

# image = generate_images(Tokenize(input_text))

# # # # Display images
# st.write("Displaying images...")
# display_images(image)

# # # if __name__ == '__main__':
# # #     st.write("Streamlit app for image generation.")
# # print("hello")