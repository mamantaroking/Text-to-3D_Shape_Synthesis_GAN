from glob import glob

import pandas as pd
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset
import numpy as np
import torch
import binvox_rw
import os
from intro import batch_size

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

'''label_mapping = {
    'chair': 0,
    'cone': 1,
    # 'die': 2,
    # 'doughnut': 3,
    # 'shoes': 4,
    # 'soccer_ball': 5
}'''

label_mapping = {
    'chair': 0,
    'shoes': 1
}

class textAndShape(Dataset):
    def __init__(self,dataframe, root_dir, mode='old'):
        # self.root_dir = root_dir
        # self.categories = os.listdir(root_dir)
        self.dataframe = dataframe
        self.mode = mode
        # self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.binvox')]
        self.files = glob(os.path.join(root_dir, '**/*.binvox'), recursive=True)
        self.labels = [label_mapping[os.path.basename(os.path.dirname(file))] for file in self.files]
        # self.wrong_descriptions = self.generate_wrong_descriptions()

        '''for category in self.categories:
            category_path = os.path.join(root_dir, category)
            category_files = glob(os.path.join(category_path, '*.binvox'))
            self.files.extend(category_files)
            self.labels.extend([label_mapping[category]] * len(category_files))'''

    '''def generate_wrong_descriptions(self):
        wrong_descriptions = []
        for i in range(len(self.dataframe)):
            wrong_idx = np.random.choice([j for j in range(len(self.dataframe)) if j != i])
            wrong_descriptions.append(self.dataframe.iloc[wrong_idx]['Descriptions'])
        return wrong_descriptions'''

    def __len__(self):
        return len(self.files) # len(self.dataframe)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(f"Current index: {idx}, DataFrame shape: {self.dataframe.shape}")

        if self.mode == 'old':


            text = self.dataframe.iloc[idx]['Descriptions']
            # text = self.dataframe.iloc[idx]
            # wrong_text = self.wrong_descriptions[idx]
            text_embedding = get_text_embedding(text).detach()  # Compute the embedding on-the-fly
            # wrong_text_embedding = get_text_embedding(wrong_text).detach()

            model = self.read_as_3d_array(self.files[idx])
            label = self.labels[idx]

            # Convert the boolean tensor to a float tensor
            model = model.astype(np.float32)
            # Add an extra dimension for channels
            model = model[np.newaxis, :]
            label_tensor = torch.tensor(label, dtype=torch.long)
            return torch.from_numpy(model), text_embedding, label_tensor  # , wrong_text_embedding
            pass


        elif self.mode == 'new':

            if idx >= len(self.dataframe):
                # Handle the out-of-bounds index, e.g., by skipping or using a default value
                print(f"Index {idx} is out of bounds. Skipping.")
                return None  # Or some default value
            else:

                file_path = self.files[idx]

                label = self.labels[idx]
                label_tensor = torch.tensor(label, dtype=torch.long)

                model = self.read_as_3d_array(file_path)
                model = model.astype(np.float32)
                model = model[np.newaxis, :]

                text = self.dataframe.iloc[idx]['Descriptions']
                # text = self.dataframe.iloc[idx]
                # wrong_text = self.wrong_descriptions[idx]
                text_embedding = get_text_embedding(text).detach()  # Compute the embedding on-the-fly
                # wrong_text_embedding = get_text_embedding(wrong_text).detach()


                return torch.from_numpy(model), text_embedding, label_tensor # , wrong_text_embedding

    def read_as_3d_array(self, file_path):
        with open(file_path, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        return model.data


# Function to get text embedding
def get_text_embedding(text):
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
        last_hidden_states = outputs.last_hidden_state
        sentence_representation = torch.mean(last_hidden_states, dim=1)
    return sentence_representation


# Read file
# df = pd.read_csv('chair_with_descriptions.csv')  # replace 'your_file.csv' with your file path
# df = pd.read_csv('training chairs_descriptions.csv')  # replace 'your_file.csv' with your file path
df = pd.read_csv('full_training_with_descriptions.csv')  # replace 'your_file.csv' with your file path
# df = pd.read_csv('chair_simple_descriptions.csv')  # replace 'your_file.csv' with your file path

# Apply BERT to each text description
df['Descriptions'] = df['Descriptions'].astype(str)
df['embedding'] = df['Descriptions'].apply(get_text_embedding)

# Create a new column for text embeddings
# df['TextEmbeddings'] = df['Descriptions'].apply(get_text_embedding)

# Save the modified DataFrame with text embeddings
df.to_csv('annotated_dataset_with_embeddings.csv', index=False)


# path = 'C:/Users/Maman/PycharmProjects/fyp/textto3dgan/binvox_training_chairs'
path = 'C:/Users/Maman/PycharmProjects/fyp/textto3dgan/full_training_set'
# path = 'C:/Users/Maman/PycharmProjects/fyp/textto3dgan/trash'
dataset = textAndShape(df, root_dir=path)
textAndShapeLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

class accuracitator(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]