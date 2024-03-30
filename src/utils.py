import os
from openai import OpenAI
from bs4 import BeautifulSoup
import opencc
import re
import numpy as np 
import pandas as pd
import json
from numpy.linalg import norm




client = OpenAI(api_key="sk-******************")

punct_regex = r"[、！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏!\"\#$%&\'\(\)\*\+,-\./:;<=>?@\[\\\]\^_`{\|}~]"

f = open("./stopwords_full.txt", "r", encoding='utf-8')
stopwords_full = f.read()
f.close()

stopwords_full = stopwords_full.strip().split('\n')

# check whether there are csv files
text_file_path = './text.csv'
if not os.path.isfile(text_file_path):
    pd.DataFrame(columns=['text', 'embedding']).to_csv(text_file_path, index=False)

question_file_path = './question.csv'
if not os.path.isfile(question_file_path):
    pd.DataFrame(columns=['question', 'embedding']).to_csv(question_file_path, index=False)


def embeddingfunc(client, text, model="text-embedding-3-small", dimensions=1536):

    response = client.embeddings.create(
    input=text,
    model=model, # released in Jan 2024 with text-embedding-3-large
    dimensions=dimensions # default 1536(small) with 3072(large)  $0.00002/token
        )
    return response.data[0].embedding


def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return '\n'.join(chunk for chunk in chunks if chunk)



def convert_to_simplified(text):
    converter = opencc.OpenCC('hk2s.json')
    simplified_text = converter.convert(text)
    return simplified_text

def rmCharacters(text, punct=True, stop=True):
    if punct == True:
        text_punct = re.sub(punct_regex, '', text) 
    if stop == True:
        pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, stopwords_full)))
        updated_text = re.sub(pattern, '', text_punct)
    return updated_text

def save2question(question, embedding): # question is string
    df = pd.read_csv(question_file_path)
    if question not in df['question'].to_list():
        pd.concat([df, pd.DataFrame({"question":[question], "embedding":[embedding]})], axis=0).to_csv(question_file_path, index=False)

def save2text(text, embedding): # text is file path
    df = pd.read_csv(text_file_path)
    if text not in df['text'].to_list():
        pd.concat([df, pd.DataFrame({"text":[text], "embedding":[embedding]})], axis=0).to_csv(text_file_path, index=False)

def simRank(question, n=3):
    question = convert_to_simplified(question)
    question = rmCharacters(question, punct=True, stop=True)
    vector = embeddingfunc(client, question, model="text-embedding-3-small", dimensions=1536)
    save2question(question, vector)
    
    textVector = pd.read_csv("./text.csv")
    B = np.array(vector)
    textVector['embedding'] = textVector['embedding'].apply(lambda x:np.array(json.loads(x)))
    textVector['sim'] = textVector['embedding'].apply(lambda x:np.dot(x,B)/(norm(x)*norm(B)))
    return textVector.sort_values(by=['sim'], ascending=False).head(n)['text']


# https://openai.com/blog/new-embedding-models-and-api-updates?ref=upstract.com
# 去除特殊字符 (https://github.com/blmoistawinde/HarvestText/tree/master)
# stopping words (https://github.com/CharyHong/Stopwords/tree/main)