#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

# In[ ]:


device = torch.device("cuda" if ... else "cpu")
print(f"Using device: {device}")

# In[ ]:


model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"We'll use the model: {model_name}")

# In[ ]:


tokenizer = .......(..., cache_dir="/tmp")
model = .......(..., cache_dir="/tmp").to(device)

# In[ ]:


texts = [
    "Artificial intelligence is revolutionizing many fields.",
    "Deep learning models require large amounts of data.",
    "PyTorch is a popular framework for machine learning.",
    "Transformers have improved natural language processing significantly.",
    "Retrieval augmented generation enhances language models with external knowledge."
]

# In[ ]:


def get_embeddings(texts, tokenizer, model):
    encoded_inputs = ...(..., padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = ...(...)
    token_embeddings = outputs.last_hidden_state
    embeddings = torch....(..., dim=1)
    return embeddings.cpu().numpy()

# In[ ]:


from sklearn.decomposition import ...

# In[ ]:


embeddings = ...

# In[ ]:


pca = ...(...)
reduced_embeddings = pca....(embeddings)

# In[ ]:


plt.figure(figsize=(5, 3))
plt.scatter(...[:, 0], ...[:, 1], marker='o', s=100)
for i, text in enumerate(texts):
    plt.annotate(text,(...[i, 0], ...[i, 1]),fontsize =5)

# # Vector database

# In[ ]:


vector_db = {
    "texts": texts,
    "embeddings": embeddings
}

# In[ ]:


def cosine_similarity(a, b):
    return ...

# In[ ]:


query = "How does PyTorch help with machine learning?"

# In[ ]:


query_embedding = get_embeddings([query], tokenizer, model)[0]

# In[ ]:


similarities = []
for i, doc_embeddings in enumerate(vector_db["embeddings"]):
    similarity = ...
    similarities.append((i, similarity, vector_db["texts"][i]))

# In[ ]:


similarities.sort(...)

# In[ ]:


print(similarities)
