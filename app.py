import streamlit as st
import torch
import json
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

# Define the model class (same as in training)
class BiLSTMPOSTag(torch.nn.Module):
    def __init__(self, token_vocab_size, token_embedding_dim,
                 pos_vocab_size, pos_embedding_dim, hidden_dim,
                 num_labels, dropout_rate):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(token_vocab_size, token_embedding_dim)
        self.pos_embeddings = torch.nn.Embedding(pos_vocab_size, pos_embedding_dim)
        self.lstm = torch.nn.LSTM(token_embedding_dim + pos_embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, tokens, pos_tags):
        token_embeds = self.token_embeddings(tokens)
        pos_embeds = self.pos_embeddings(pos_tags)
        combined = torch.cat((token_embeds, pos_embeds), dim=2)
        lstm_out, _ = self.lstm(combined)
        output = self.dropout(lstm_out)
        logits = self.classifier(output)
        return logits

# Load vocab files
with open("token_vocab.json") as f:
    TOKEN_VOCAB = json.load(f)
with open("pos_vocab.json") as f:
    POS_VOCAB = json.load(f)
with open("id_to_label.json") as f:
    ID_TO_LABEL = json.load(f)
ID_TO_LABEL = {int(k): v for k, v in ID_TO_LABEL.items()}  # Convert keys to int

# Load model
model = BiLSTMPOSTag(
    token_vocab_size=len(TOKEN_VOCAB),
    token_embedding_dim=100,
    pos_vocab_size=len(POS_VOCAB),
    pos_embedding_dim=25,
    hidden_dim=128,
    num_labels=len(ID_TO_LABEL),
    dropout_rate=0.33,
)
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"), strict=False)
model.eval()

def get_pos_tags(tokens):
    doc = nlp(" ".join(tokens))
    return [token.pos_ for token in doc]

# Preprocess function
def encode_input(tokens, pos_tags):
    token_ids = [TOKEN_VOCAB.get(token.lower(), TOKEN_VOCAB.get("<unk>", 0)) for token in tokens]
    pos_ids = [POS_VOCAB.get(tag, POS_VOCAB.get("<unk>", 0)) for tag in pos_tags]
    token_tensor = torch.tensor([token_ids])
    pos_tensor = torch.tensor([pos_ids])
    return token_tensor, pos_tensor

# Streamlit UI
st.title("BiLSTM POS Tagging")
sentence = st.text_input("Enter a sentence:")

if sentence:
    tokens = sentence.strip().split()
    pos_tags = get_pos_tags(tokens)   # Replace with real POS tagger

    token_tensor, pos_tensor = encode_input(tokens, pos_tags)

    with torch.no_grad():
        logits = model(token_tensor, pos_tensor)
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    labels = [ID_TO_LABEL[idx] for idx in predictions]

    st.subheader("Predicted Labels:")
    for token, label in zip(tokens, labels):
        st.write(f"**{token}** → {label}")
