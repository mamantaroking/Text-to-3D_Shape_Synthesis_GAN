import torch
import torch.nn as nn
import numpy as np
import os
from torchvision.utils import save_image
import plotly.graph_objects as go
from SANITY_CHECK import ShowVoxelModel
from new_texter_model7 import textG, device, nz
# from textto3dgan.preprocessing import textAndShapeLoader, textAndShape
# from textto3dgan.intro import  batch_size
from transformers import BertModel, BertTokenizer
from pytorch3d.ops import cubify
from pytorch3d.io import save_obj
import streamlit as st
from FIDType2 import calculate_fid

model_path = 'C:/Users/Maman/PycharmProjects/fyp/textto3dgan/model_save/models_inCaptive/G.pth'
textG.load_state_dict(torch.load(model_path))
image_path = PATH = 'C:/Users/Maman/PycharmProjects/fyp/textto3dgan/model_save/'


# text_caption = 'Nike Air Force'
# text_caption = 'Leather lounge chair with holes'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_text_embedding(text):
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
        last_hidden_states = outputs.last_hidden_state
        sentence_representation = torch.mean(last_hidden_states, dim=1)
    return sentence_representation


# text_embedding = get_text_embedding(text_caption).detach()


def synthesizer():
    print('Starting to generate images from text descriptions.')
    textG.eval()  # Set the generator to evaluation mode

    with torch.no_grad():  # No gradients required for inference
        # for i, (voxel_files, _, _) in enumerate(textAndShapeLoader):
        for i in range(1):
            # Generate random noise vector
            random_noise = torch.randn(1, 512, 1, 1, 1, device=device)
            random_noise = random_noise.to(device)
            # voxel_files = voxel_files.to(device)
            text_embeddings = text_embedding.to(device)

            # Generate images using the generator model
            generated_images = textG(random_noise, text_embeddings) # .detach()
            dump_dir = os.path.join(PATH, 'images_generated_from_text.')

            # Create the directory if it doesn't exist
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)

            samples = generated_images.cpu().detach().squeeze(dim=1).numpy()
            out = generated_images.cpu().detach().squeeze(dim=1)
            print(samples.shape)

            image_saved_path = os.path.join(PATH + 'produce/images')
            ShowVoxelModel(samples, image_saved_path, i)

            threshold = 0.8
            meshes = cubify(out, threshold)

            verts = meshes.verts_packed().cpu().numpy()
            faces = meshes.faces_packed().cpu().numpy()

            vertixes = meshes.verts_packed()
            facixes = meshes.faces_packed()



            save_obj("model_save/produce/mesh/mesh3.obj", vertixes, facixes)

            print(f'Batch {i} - Finished generating images from text descriptions.')




def get_voxel_and_embeddings_batch(text_embedding, dataset, batch_size=64, nz=768):
    # Check if the dataset is an instance of a dataset class
    if not hasattr(dataset, '__len__'):
        raise ValueError("The 'dataset' argument must be a dataset object with a __len__ method.")

    # Initialize arrays to hold the batch data
    # embeddings = torch.zeros((batch_size, nz, 1, 1, 1))
    embeddings = torch.zeros((batch_size, text_embedding, 1, 1, 1))
    voxel_files = []

    # Randomly select indices for the batch
    batch_indices = torch.randint(0, len(dataset), (batch_size,))

    for idx, data_idx in enumerate(batch_indices):
        # Retrieve the voxel file and its corresponding embedding
        voxel_file, text_embedding = dataset[data_idx]

        # Add the retrieved data to the batch arrays
        embeddings[idx] = text_embedding
        voxel_files.append(voxel_file)

    return voxel_files, embeddings


# if __name__ == '__main__':
    # synthesizer()

if __name__ == '__main__':
    st.title("Text-To-3D GAN")
    st.write("")

    st.divider()
    st.header("Text Input")
    # st.markdown("***")
    # st.text("")
    # st.write("")

    if st.text_input("Input Text Prompt:", key="Prompt"):
        text_prompt = st.session_state.Prompt
        st.write(text_prompt)
        text_embedding = get_text_embedding(text_prompt).detach()
    # You can access the value at any point with:
    # st.text_input("Input Text Prompt:", key="Prompt")
    st.write("")

    if st.button('Run'):
        synthesizer()


    st.divider()
    st.header("3D Output")
    # st.markdown("***")

    # st.image('mesh2.png', caption=st.session_state.Prompt)

    if os.path.exists("model_save/produce/mesh/mesh3.obj"):
        with open("model_save/produce/mesh/mesh3.obj", "rb") as file:
            btn = st.download_button(
                label="Download .obj file",
                data=file,
                file_name=st.session_state.Prompt + ".obj",
                mime="3D/.obj"
            )