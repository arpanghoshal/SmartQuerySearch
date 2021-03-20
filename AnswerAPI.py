"""
Imports
"""


#flask
from flask import Flask,jsonify,request,make_response
from urllib import parse

#wiki name
import requests
from bs4 import BeautifulSoup
import re

#wiki parse

#semantic search
from sentence_transformers import SentenceTransformer
import scipy

#summarizer
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

from keras import models   

import os
import spacy

from pysbd.utils import PySBDFactory

app = Flask(__name__)




model1 = torch.load('models/SentenceTransformer.pt')
model2 = torch.load('models/T5ForConditionalGeneration.pt')
tokenizer = torch.load('models/T5Tokenizer.pt')
device = torch.device('cpu')



"""
Functions
"""

#Wiki name
def wikiname(query,website):
  search = query+ ':' +website
  results = 1
  page = requests.get(f"https://www.google.com/search?q={search}&num={results}")
  soup = BeautifulSoup(page.content, "lxml")
  links = soup.findAll("a")
  for link in links :
    link_href = link.get('href')
    if "url?q=" in link_href and not "webcache" in link_href:
      link =  link.get('href').split("?q=")[1].split("&sa=U")[0]
      break

  return link

def retrieve(link):
  #os.system('pip install trafilatura')
  os.system(("trafilatura -u '{0}' > retrieve.txt").format(link))

def deletesen():
  file1 = open("retrieve.txt","r+")
  content=file1.read()

  nlp = spacy.blank('en')
  nlp.add_pipe(PySBDFactory(nlp))

  doc = nlp(content)
  lala=list(doc.sents)
  ll=len(lala)
  for i in range(ll):
    lala[i]=str(lala[i])
  i=0
  while(i<len(lala)):
    count=0
    for j in lala[i].split():
      if(j.isalpha()):
        count+=1
    if(count<5):
      del lala[i]
    else:
      i+=1
  return lala
  
def semsearch(ForSemSrch,query):
  sentence_embeddings = model1.encode(ForSemSrch)

  queries = [query]
  query_embeddings = model1.encode(queries)

  number_top_matches = 10

  for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    li=[]
    for idx, distance in results[0:number_top_matches]:
      ToBeSumm = ForSemSrch[idx].strip()
      li.append(ToBeSumm)
  mainstr = ''.join(map(str, li))
  
  return mainstr
  
def summ(ToBeSumm):

  preprocess_text = ToBeSumm.strip().replace("\n","")
  t5_prepared_Text = "summarize: "+preprocess_text

  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt")    # do baar kiya hua hai


  # summmarize 
  summary_ids = model2.generate(tokenized_text,
                                      num_beams=1, length_penalty=1.0,
                                      no_repeat_ngram_size=2,
                                      min_length=5,
                                      max_length=60,
                                      early_stopping=False)

  output = tokenizer.decode(summary_ids[0], skip_special_tokens=False)

  return output


""" 
Flask decorator 
"""

@app.route('/answer/<website>/',methods=['GET'])
def query(website):
	query = request.args.get('query')
	query = parse.unquote(query)
	
	link = wikiname(query,website)
	retrieve(link)
	corpus = deletesen()
	ToBeSumm = semsearch(corpus,query)
	OurFinalOutput = summ(ToBeSumm)
	

	return jsonify(OurFinalOutput)


""" 
Call 
"""

if __name__=='__main__':
	app.run(debug=True,host="0.0.0.0",port=5052)


		
