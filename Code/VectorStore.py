# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 08:39:18 2024

@author: Trevor Gratz, trevormgratz@gmail.com

The following code implements a class to embed and save a vector store of 
Pierce County's homeless program policies.

To do: 
    1) Remove filler from PDFs
    2) Splitting PDF on pages means there is no overlap in some sections
    2) Research other embedding models
        - Try https://huggingface.co/dunzhang/stella_en_400M_v5 
    3) How many chunks to return?
    4) How large should the context window be?
    3) The word documets/Appendices are forms. This will neeed special handling.
    
https://www.piercecountywa.gov/7587/Homeless-Program-Policies

"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS


class VectorStore():
    def __init__(self, embed_mod='BAAI/bge-small-en-v1.5'):
        
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        self.embedding_mod = HuggingFaceBgeEmbeddings(
            model_name=embed_mod,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
    def loadDocuments(self, path, doctype='PDF'):
        '''
        Load the PDFs using Langchain's PDF Directory Loader (see imported package above)
        
        Args:
            data_dir: String path of folder location of PDFs to load

        Returns:
            documents: list of langchain documents
        '''
        if doctype == 'PDF':
            documents = PyPDFLoader(path).load()
            
        elif doctype == "Word":
            pass
        
        return documents
        
    def removeIntro(self, documents, thresh=0.08, charlist=['\n', '.']):
        '''
        Removes pages containing specific characters above a certain threshold
        
        Parameters
        ----------
        p : TYPE List of LangChain Documents
            DESCRIPTION.

        Returns
        -------
        doc : List of LangChain Documents
            DESCRIPTION. Filters the original list to remove intro pages

        '''
        docs = []
        
        for p in documents:
            characters = list((p.page_content))
            counter = 0 
            # count the number of user defined character
            for c in charlist:
                counter += characters.count(c)
            
            per = counter/len(characters)
        
            # Maintain documents below a threshold
            if per < thresh:
                docs.append(p)
            
        return docs
    
    def normalizeDocuments(self, documents):
        '''
        Normalizes data using Autotokenizer
        
        Args:
            documents: list of langchain documents
          
        
        Returns:
            docs: list of langchain documents
        '''
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        docs = []
        for d in documents:
            normtext = tokenizer.backend_tokenizer.normalizer.normalize_str(d.page_content)
            d.page_content=normtext
            docs.append(d)
        return docs 
    
    def splitDocuments(self, documents, chunk_size=700, chunk_overlap=50):
        '''
        Split the loaded documents into smaller chunks. 
        
        Args:
            documents: list of langchain documents
            chunk_size: int, number of characters in a chunk
            chunk_overlap: int, number of characters overlapping between adjacent chunks
        
        Returns:
            document_chunks: list of langchain document chunks
        '''
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )
        
        document_chunks = text_splitter.split_documents(documents)
        return document_chunks
        

    def buildVectorStore(self, document_chunks):
        '''

        Args:
            document_chunks: list of langchain document chunks
            

        Returns:
            vectorstore: langchain VectorStore
        '''

        vectorstore = FAISS.from_documents(document_chunks, self.embedding_mod)
        return vectorstore