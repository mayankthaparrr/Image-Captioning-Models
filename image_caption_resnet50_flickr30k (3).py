# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T10:42:31.130166Z","iopub.execute_input":"2026-06-29T10:42:31.131016Z","iopub.status.idle":"2026-06-29T10:42:48.497381Z","shell.execute_reply.started":"2026-06-29T10:42:31.130981Z","shell.execute_reply":"2026-06-29T10:42:48.496709Z"}}
import os
import torch
import torch.nn as nn
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.models import ResNet50_Weights
import spacy

spacy_eng = spacy.load("en_core_web_sm")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU count: {torch.cuda.device_count()}")

# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T10:42:48.498711Z","iopub.execute_input":"2026-06-29T10:42:48.499268Z","iopub.status.idle":"2026-06-29T10:42:48.506527Z","shell.execute_reply.started":"2026-06-29T10:42:48.499241Z","shell.execute_reply":"2026-06-29T10:42:48.505941Z"}}
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sent_list):
        freqs = {}
        idx = 4
        for sent in sent_list:
            sent = str(sent)
            for word in self.tokenize(sent):
                if word not in freqs:
                    freqs[word] = 1
                else:
                    freqs[word] += 1
                if freqs[word] == self.freq_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self, sents):
        tokens = self.tokenize(sents)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in tokens]

# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T10:42:48.508028Z","iopub.execute_input":"2026-06-29T10:42:48.508253Z","iopub.status.idle":"2026-06-29T10:42:48.526550Z","shell.execute_reply.started":"2026-06-29T10:42:48.508231Z","shell.execute_reply":"2026-06-29T10:42:48.525968Z"}}
class FlickrDataset(Dataset):
    def __init__(self, root_dir, caps, transforms=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(caps, delimiter='|')
        self.transforms = transforms

        # drop any rows where caption is missing
        self.df = self.df.dropna(subset=[' comment'])
        self.df = self.df.reset_index(drop=True)

        self.img_pts = self.df['image_name']
        self.caps = self.df[' comment']

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.caps.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = str(self.caps[idx])
        img_pt = self.img_pts[idx]
        img = Image.open(os.path.join(self.root_dir, img_pt)).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        numberized_caps = []
        numberized_caps += [self.vocab.stoi["<SOS>"]]
        numberized_caps += self.vocab.numericalize(caption)
        numberized_caps += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(numberized_caps), img_pt

# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T10:42:48.528372Z","iopub.execute_input":"2026-06-29T10:42:48.528575Z","iopub.status.idle":"2026-06-29T10:42:48.544196Z","shell.execute_reply.started":"2026-06-29T10:42:48.528554Z","shell.execute_reply":"2026-06-29T10:42:48.543571Z"}}
class CapCollat:
    def __init__(self, pad_seq, batch_first=False):
        self.pad_seq = pad_seq
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [itm[0].unsqueeze(0) for itm in batch]
        imgs = torch.cat(imgs, dim=0)
        target_caps = [itm[1] for itm in batch]
        target_caps = pad_sequence(target_caps, batch_first=self.batch_first,
                                   padding_value=self.pad_seq)
        img_names = [itm[2] for itm in batch]
        return imgs, target_caps, img_names

# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T10:42:48.545005Z","iopub.execute_input":"2026-06-29T10:42:48.545209Z","iopub.status.idle":"2026-06-29T10:43:11.892532Z","shell.execute_reply.started":"2026-06-29T10:42:48.545171Z","shell.execute_reply":"2026-06-29T10:43:11.891841Z"}}
ROOT_DIR  = '/kaggle/input/datasets/hsankesara/flickr-image-dataset/flickr30k_images/flickr30k_images'
CAPS_FILE = '/kaggle/input/datasets/hsankesara/flickr-image-dataset/flickr30k_images/results.csv'

dataset = FlickrDataset(
    root_dir=ROOT_DIR,
    caps=CAPS_FILE,
    transforms=T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=CapCollat(pad_seq=dataset.vocab.stoi['<PAD>'])
)

print(f"Dataset size: {len(dataset)}")
print(f"Vocab size:   {len(dataset.vocab)}")

# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T10:43:11.893429Z","iopub.execute_input":"2026-06-29T10:43:11.893762Z","iopub.status.idle":"2026-06-29T11:13:21.265657Z","shell.execute_reply.started":"2026-06-29T10:43:11.893738Z","shell.execute_reply":"2026-06-29T11:13:21.264943Z"}}
# ── FEATURE EXTRACTION ───────────────────────────────────────────
# Run once — saves features to pkl so you don't re-run ResNet every session

FEATURES_FILE = 'flickr30k_features.pkl'

if os.path.exists(FEATURES_FILE):
    print("Loading features from pickle...")
    with open(FEATURES_FILE, 'rb') as f:
        image_features = pickle.load(f)
    print(f"Loaded features for {len(image_features)} images")

else:
    print("Extracting features with ResNet50...")
    weights = ResNet50_Weights.DEFAULT
    resnet  = models.resnet50(weights=weights)
    resnet  = nn.Sequential(*list(resnet.children())[:-1])
    resnet  = resnet.to(device)
    resnet.eval()

    image_features = {}
    with torch.no_grad():
        for batch_idx, (images, captions, img_names) in enumerate(dataloader):
            images = images.to(device)
            feats  = resnet(images)
            feats  = feats.view(feats.size(0), -1)  # (batch, 2048)
            for i, img_name in enumerate(img_names):
                image_features[img_name] = feats[i].cpu().numpy()
            if batch_idx % 100 == 0:
                print(f"  Processed {batch_idx * 32} images...")

    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(image_features, f)
    print(f"Extracted and saved features for {len(image_features)} images")

# verify
sample_key = list(image_features.keys())[0]
print(f"Sample: {sample_key} → shape {image_features[sample_key].shape}")

# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T11:13:21.266594Z","iopub.execute_input":"2026-06-29T11:13:21.267009Z","iopub.status.idle":"2026-06-29T11:13:21.276418Z","shell.execute_reply.started":"2026-06-29T11:13:21.266983Z","shell.execute_reply":"2026-06-29T11:13:21.275664Z"}}
# ── ENCODER ──────────────────────────────────────────────────────
# Takes 2048-d ResNet features, projects to hidden_size
# Outputs initial hidden + cell state for LSTM

class Encoder(nn.Module):
    def __init__(self, image_feature_size, hidden_size):
        super().__init__()
        self.fc1     = nn.Linear(image_feature_size, hidden_size)
        self.ln1     = nn.LayerNorm(hidden_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, image_features):
        h = self.dropout(self.relu(self.ln1(self.fc1(image_features))))
        c = self.dropout(self.relu(self.ln1(self.fc1(image_features))))
        return h.unsqueeze(0), c.unsqueeze(0)


# ── DECODER ──────────────────────────────────────────────────────
# LSTM that generates caption word by word

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embed_ln  = nn.LayerNorm(embedding_size)
        self.lstm      = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(hidden_size, vocab_size)

    def forward(self, captions, hidden, cell):
        embeddings = self.embed_ln(self.embedding(captions))
        outputs, (hidden, cell) = self.lstm(embeddings, (hidden, cell))
        output = self.fc(self.dropout(outputs))
        return output, hidden, cell


# ── SEQ2SEQ ──────────────────────────────────────────────────────
# Wraps encoder + decoder together

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self.encoder = Encoder(2048, hidden_size)
        self.decoder = Decoder(vocab_size, embedding_size, hidden_size)

    def forward(self, image_features, captions):
        hidden, cell = self.encoder(image_features)
        outputs, _, _ = self.decoder(captions, hidden, cell)
        return outputs

print("Model classes defined.")

# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T11:13:21.277304Z","iopub.execute_input":"2026-06-29T11:13:21.277496Z","iopub.status.idle":"2026-06-29T11:13:26.863170Z","shell.execute_reply.started":"2026-06-29T11:13:21.277476Z","shell.execute_reply":"2026-06-29T11:13:26.862284Z"}}
# ── TRAINING DATASET ─────────────────────────────────────────────
# Rebuilds dataset using pre-extracted features instead of raw images

class FeatureDataset(Dataset):
    def __init__(self, features_dict, df, vocab):
        self.features_dict = features_dict
        self.vocab = vocab
        self.samples = []
        for _, row in df.iterrows():
            img_name = row['image_name']
            caption  = str(row[' comment'])
            if img_name in features_dict:
                self.samples.append((img_name, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        features = torch.FloatTensor(self.features_dict[img_name])
        tokens   = [self.vocab.stoi['<SOS>']]
        tokens  += self.vocab.numericalize(caption)
        tokens  += [self.vocab.stoi['<EOS>']]
        return features, torch.LongTensor(tokens)

def feature_collate(batch):
    features, captions = zip(*batch)
    features = torch.stack(features, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return features, captions

# split 80/10/10
all_imgs = list(image_features.keys())
random.seed(42)
random.shuffle(all_imgs)

n = len(all_imgs)
train_imgs = set(all_imgs[:int(0.8 * n)])
val_imgs   = set(all_imgs[int(0.8 * n):int(0.9 * n)])
test_imgs  = set(all_imgs[int(0.9 * n):])

train_df = dataset.df[dataset.df['image_name'].isin(train_imgs)].reset_index(drop=True)
val_df   = dataset.df[dataset.df['image_name'].isin(val_imgs)].reset_index(drop=True)
test_df  = dataset.df[dataset.df['image_name'].isin(test_imgs)].reset_index(drop=True)

train_dataset = FeatureDataset(image_features, train_df, dataset.vocab)
val_dataset   = FeatureDataset(image_features, val_df,   dataset.vocab)
test_dataset  = FeatureDataset(image_features, test_df,  dataset.vocab)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  collate_fn=feature_collate)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, collate_fn=feature_collate)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T11:13:26.865261Z","iopub.execute_input":"2026-06-29T11:13:26.865546Z","iopub.status.idle":"2026-06-29T11:40:34.218874Z","shell.execute_reply.started":"2026-06-29T11:13:26.865522Z","shell.execute_reply":"2026-06-29T11:40:34.217989Z"}}
# ── HYPERPARAMETERS ──────────────────────────────────────────────
VOCAB_SIZE     = len(dataset.vocab)
EMBEDDING_SIZE = 512
HIDDEN_SIZE    = 512
NUM_EPOCHS     = 20
LR             = 0.001

model     = Seq2Seq(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_losses   = []
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # ── TRAIN ──
    model.train()
    total_loss = 0
    for features, captions in train_loader:
        features = features.to(device)
        captions = captions.to(device)
        inputs  = captions[:, :-1]   # everything except <EOS>
        targets = captions[:, 1:]    # everything except <SOS>
        outputs = model(features, inputs)
        loss = criterion(outputs.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # ── VALIDATE ──
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for features, captions in val_loader:
            features = features.to(device)
            captions = captions.to(device)
            inputs  = captions[:, :-1]
            targets = captions[:, 1:]
            outputs = model(features, inputs)
            loss = criterion(outputs.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            val_loss += loss.item()

    avg_train = total_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    train_losses.append(avg_train)
    val_losses.append(avg_val)

    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ✓ Saved best model (val loss: {avg_val:.4f})")

# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T11:40:34.219798Z","iopub.execute_input":"2026-06-29T11:40:34.220052Z","iopub.status.idle":"2026-06-29T11:40:34.556645Z","shell.execute_reply.started":"2026-06-29T11:40:34.220028Z","shell.execute_reply":"2026-06-29T11:40:34.555819Z"}}
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses)+1), train_losses, 'b-o', label='Train Loss')
plt.plot(range(1, len(val_losses)+1),   val_losses,   'r-o', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T11:40:34.557585Z","iopub.execute_input":"2026-06-29T11:40:34.558246Z","iopub.status.idle":"2026-06-29T11:40:34.843384Z","shell.execute_reply.started":"2026-06-29T11:40:34.558207Z","shell.execute_reply":"2026-06-29T11:40:34.842747Z"}}
def greedy_caption(model, image_feature, vocab, max_len=20):
    model.eval()
    idx2word = vocab.itos
    with torch.no_grad():
        feat   = torch.FloatTensor(image_feature).unsqueeze(0).to(device)
        hidden, cell = model.encoder(feat)
        word   = torch.LongTensor([[vocab.stoi['<SOS>']]]).to(device)
        caption = []
        for _ in range(max_len):
            out, hidden, cell = model.decoder(word, hidden, cell)
            word_id = out.argmax(dim=2).item()
            if word_id == vocab.stoi['<EOS>']:
                break
            caption.append(idx2word[word_id])
            word = torch.LongTensor([[word_id]]).to(device)
    return ' '.join(caption)

# test on 5 random images from test set
model.load_state_dict(torch.load('best_model.pth'))
test_img_list = list(test_imgs)[:5]

for img_name in test_img_list:
    if img_name not in image_features:
        continue
    caption = greedy_caption(model, image_features[img_name], dataset.vocab)
    print(f"Image:   {img_name}")
    print(f"Caption: {caption}")
    print("-" * 50)

# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T11:40:34.844267Z","iopub.execute_input":"2026-06-29T11:40:34.844653Z","iopub.status.idle":"2026-06-29T11:41:39.940493Z","shell.execute_reply.started":"2026-06-29T11:40:34.844627Z","shell.execute_reply":"2026-06-29T11:41:39.939815Z"}}
# ── EVALUATION ───────────────────────────────────────────────────
# Run this after training to get BLEU, Precision, Recall, F1

import nltk
from nltk.translate.bleu_score import corpus_bleu
from collections import Counter

nltk.download('punkt', quiet=True)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

references  = []
hypotheses  = []

print("Generating captions for test set...")
for img_name in list(test_imgs):
    if img_name not in image_features:
        continue

    # generated caption
    generated = greedy_caption(model, image_features[img_name], dataset.vocab)
    hyp = generated.split()

    # ground truth captions for this image (all 5)
    gt_rows = dataset.df[dataset.df['image_name'] == img_name][' comment']
    ref = [str(cap).split() for cap in gt_rows.tolist()]

    references.append(ref)
    hypotheses.append(hyp)

# ── BLEU ──────────────────────────────────────────────────────────
bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
bleu4 = corpus_bleu(references, hypotheses)

# ── PRECISION / RECALL / F1 ───────────────────────────────────────
all_ref_tokens = []
all_hyp_tokens = []
for ref_list, hyp in zip(references, hypotheses):
    all_ref_tokens.extend(ref_list[0])
    all_hyp_tokens.extend(hyp)

ref_counter = Counter(all_ref_tokens)
hyp_counter = Counter(all_hyp_tokens)
tp        = sum((hyp_counter & ref_counter).values())
precision = tp / sum(hyp_counter.values()) if sum(hyp_counter.values()) > 0 else 0
recall    = tp / sum(ref_counter.values()) if sum(ref_counter.values()) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'='*40}")
print(f"BLEU-1:    {bleu1:.4f}")
print(f"BLEU-4:    {bleu4:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"{'='*40}")

# %% [code]
