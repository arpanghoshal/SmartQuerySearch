#semantic search
from sentence_transformers import SentenceTransformer
import scipy

#summarizer
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


model1 = SentenceTransformer('distilroberta-base-msmarco-v2') 

model2 = T5ForConditionalGeneration.from_pretrained('t5-small') 
  
tokenizer = T5Tokenizer.from_pretrained('t5-small')

device = torch.device('cpu')

torch.save(model1,'models/SentenceTransformer.pt')
torch.save(model2,'models/T5ForConditionalGeneration.pt')
torch.save(tokenizer,'models/T5Tokenizer.pt')
