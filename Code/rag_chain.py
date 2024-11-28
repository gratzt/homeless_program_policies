# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 10:28:03 2024

@author: Trevor Gratz, trevormgratz@gmail.com

The following code utilizes the users query to retrieve infomration from
Pierce County's homeless program policies and return relevant chunks of 
text from the manual. 
"""

import os
from dotenv import load_dotenv  # used to load the huggingface-api-key
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate


class RAG_Chain:
    def __init__(self, embed_mod = r'BAAI/bge-small-en-v1.5'): 
        '''
        Load the api-key from the .env file and initialize self.llm and self.retriever_system

        Initialize:
            self.llm: LLM from HuggingFaceHub.
            self.retriever_system: your implemented retrieval system from 
                                   retriever.py

        Returns:
            None

        '''
        # load the api-key - do not change
        result = load_dotenv()
        api_key = os.getenv('HUGGINGFACE_API_KEY')
        

        # load the LLM
        self.llm = HuggingFaceHub(
            repo_id='google/flan-t5-small',
            model_kwargs={"temperature": 0.0},
            huggingfacehub_api_token=api_key
        )
        
        model_name = embed_mod
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        self.embedding_mod = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        self.vectorstore = FAISS.load_local(
            r"..\Vector_DB\faiss_index_homeless_policy",
            self.embedding_mod,
            allow_dangerous_deserialization=True
        )
        
        self.retriever = self.vectorstore.as_retriever()
        
    def format_docs(self, docs):
        outstr = '\n'
        for doc in docs:
            outstr += doc.metadata['source'] + '\n' +'\n' + doc.page_content + '\n' +'\n'
        return outstr
        #return "\n\n".join(doc.page_content for doc in docs)
    
    def custom_prompt(self, info):
        prompt = "Summarize the following pieces of retrieved context, that answer this question.\n" + f"Question: {info['question']} \n Context: {info['context']}"
        return prompt

    def createRAGChain(self):
        '''
        NOTE: The retrieveal is often working, howerever, the LLM is doing 
        a poor job summarizing the context. Even when explicitly prompted to
        summarize. For now just utilize retrieveal, but revisit with another
        llm.
        
        Build the RAG pipeline 
        Args:
            None
            
        Returns:
            qa_chain: LangChain.
        '''
        #prompt = hub.pull("rlm/rag-prompt")
        prompt = PromptTemplate.from_template('''The following pieces of retrieved context may answer your question.
                                              Question: {question}
                                              Context: {context}''')
        qa_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            #| self.llm
            #| StrOutputParser()
            )
        return qa_chain
