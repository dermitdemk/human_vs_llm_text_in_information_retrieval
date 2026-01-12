from rank_bm25 import BM25Okapi
from typing import Literal
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import faiss
import numpy as np
from tqdm import tqdm

def bm25_implement(query:list[str],n:int,bm25,df)->pd.DataFrame:
    """
    Makes a query and returns the number of correct matches in the top k

    Args:
        query (list of str): Query terms 
        n (int): How many documents retrieved
        bm25: BM25 model 
        df (pd.DataFrame): DataFrame 

    Return:
        pd.DataFrame: a dataframe with the top n articles for that query
    """
    #df_tags = df[['tags']].copy()
    df_with_score = df.copy()
    if isinstance(query,list) == False:
        raise ValueError ('query must be a list of strs')
    doc_scores = bm25.get_scores(query)
    df_with_score['score'] = doc_scores
    top_n = df_with_score.sort_values('score',ascending=False).iloc[:n]
    
    return top_n


def bm25(modus:Literal['content','summary','content_and_summary','llm_text'],
         querrys:list[str],n:int,df:pd.DataFrame)->list:
    """
    Run BM25 retrieval for multiple queries on a selected text field of the dataframe.

    Args:
    - modus (Literal): Text source to use for retrieval
    - querrys (list with str): List of queries
    - n (int): Number of top documents to retrieve per query
    - df (pd.DataFrame): Dataframe containing the text

    Returns:
    - list: A list of lists, where the inner list contains the dataframe indices
            of the top retrieved documents for the corresponding query
    """
    

    if modus == 'content':
        courpus = df['content']
    if modus == 'summary':
        courpus = df['summary']
    if modus == 'content_and_summary':
        courpus =  df['content'] + df['summary']
    if modus == 'llm_text':
        courpus = df['llm_text']

        
    tokenized_corpus = [doc.split(" ") for doc in courpus]
    bm25_model = BM25Okapi(tokenized_corpus)
    
    query_results = []
    for q in querrys:
        if not isinstance(q,list):
            ## split querry into tokens
            q = q.split(" ")
        index = bm25_implement(q,n,bm25_model,df).index
        query_results.append(list(index))
    return query_results
        
        

######################################################################
## bi encoder




            
def bi_encoder(modus:Literal['content','summary','content_and_summary','llm_text'],
         querrys:list[str],n:int,df:pd.DataFrame)->list:
    """
    Retrieve and rerank documents for multiple queries using a bi-encoder + cross-encoder
    pipeline.

    Args:
    - modus (Literal): Text source to use for retrieval
    - querrys (list with str): List of queries
    - n (int): Number of top documents to retrieve per query
    - df (pd.DataFrame): Dataframe containing the text

    Returns:
    - list: A list of lists, where the inner list contains the dataframe indices
            of the top retrieved documents for the corresponding query
"""

    if modus == 'content':
        courpus = df['content']
    if modus == 'summary':
        courpus = df['summary']
    if modus == 'content_and_summary':
        courpus =  df['content'] + df['summary']
    if modus == 'llm_text':
        courpus = df['llm_text']
    #test = 'test'
    reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',device='cpu')
    
    embeddings, model = creat_document_embeddings(corpus=courpus)
    querry_results = []
    for q in tqdm(querrys):
        if isinstance(q,list):
            q = " ".join(q)
        if not isinstance(q,str):
            q = str(q)
            
        querry_embedding = embed_querry(q,model)
        index = get_k_kanidates(querry_embedding,n*3,embeddings)
        selected_articles = df.iloc[index[0]].copy()
        found_articles = rerank_the_articles(selected_articles,n,q,reranker)
        querry_results.append(list(found_articles.index))
    return querry_results
        
        
    
    
    
    

def creat_document_embeddings(corpus):
    """
    Create document embeddings and a faiss index from the text corpus.

    Args:
    - corpus: Collection of texts

    Returns:
    - tuple: faiss Index and SentenceTransformer
    """
    model = SentenceTransformer('intfloat/multilingual-e5-small',device='cpu')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to('cpu')
    courpus_embeddings = model.encode(corpus)
    embedding_dimension = courpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(courpus_embeddings)
    return index, model

def embed_querry(querry,model):
        # embeds the query with the same model es the coprus
        return model.encode([querry])

def get_k_kanidates(querry_embedding,k,embeddings):
        ## retrieve k canidates which then get reranked
        #querry_embedding = embed_querry(querry)
        distances, index = embeddings.search(querry_embedding,k)
        return index
def rerank_the_articles (selected_articles,n,querry,reranker):
    ## cross encodes the selected articles and ranks them
    
    reranker.to('cpu')
    pairs =[[querry,doc]for doc in selected_articles['content']]
    scores = reranker.predict(pairs)
    selected_articles['scores'] = scores
    return selected_articles.sort_values('scores',ascending=False).iloc[:n]



##### Evaluation #################ü
def position_of_article(results):
    """
    Gets the resulting dataframe and returns 2 lists, which are the possition of the query in the content column and the llm colum

    Args:
    - results (_type_): _description_
    
    Returns:
    - tupl(llm_postition, content_possiton): The position of the article in the llm and the content query
    """
    def pos_in_list(l,q):
        if q not in l:
            return -1
        return l.index(q) +1

    llm_pos = []
    content_pos = []
    for i ,row in results.iterrows():
        llm = row['llm_text']
        if isinstance(llm,str):
            llm = eval(llm)
        content = row['content']
        if isinstance(content,str):
            content = eval(content)
        llm_pos.append(pos_in_list(llm,i))
        content_pos.append(pos_in_list(content,i))
        
    return (llm_pos, content_pos)

def number_of_tags_match(result,normelized=False):
    """
    Compute the number of matching tags between queries and retrieved document sets.

    Args:
    - result (pd.DataFrame): The dataframe 
    - normelized (bool): If True, matches are weighted by inverse tag frequency.
                         If False, simple match counts are used. (The default is false.)

    Returns:
    - tuple[list[float], list[float]]: The first list shows match scores for text retrieved via "llm_text" 
                                       and the second list shows match scores for documents retrieved via "content".
    """
    if isinstance(result.iloc[0]['querry'],str):
        result['querry'] = [eval(q) for q in result['querry']]
    tag_lookup = {i:t for i,t in zip(result.index,result['querry'])}
        
    if normelized == True:
        all_tags = pd.Series([i for l in result['querry'] for i in  l])
        frequency_of_tags = all_tags.value_counts()
    
    def get_num_of_matches(l:list[int],q:list[str],normelize=False):
        number_of_matches = 0
        for i in l:
            tags = tag_lookup[i]
            match = set(tags).intersection(set(q))
            # Tags that occur more frequently count less, rare tags count more
            if normelized == True:
                addition_factor = sum([1/frequency_of_tags[m] for m in match])
                
                
            if normelize ==False :
                addition_factor = len(match)
            number_of_matches += addition_factor
        return number_of_matches 
            
    llm_mean_matches=[]
    content_mean_matches=[]
    for i,row in result.iterrows():
        querry = row['querry']
        if isinstance(querry,str):
            querry = eval(querry)
        content = row['content']
        if isinstance(content,str):
            content = eval(content)
        llm = row['llm_text']
        if isinstance(llm,str):
            llm = eval(llm)
        llm_mean_matches.append(get_num_of_matches(llm,querry,normelize=normelized))
        content_mean_matches.append(get_num_of_matches(content,querry,normelize=normelized))
    return (llm_mean_matches, content_mean_matches)
        
