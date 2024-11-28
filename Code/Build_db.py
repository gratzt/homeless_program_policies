# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 08:00:57 2024

@author: Trevor Gratz, trevormgratz@gmail.com

This file loads in PDFs on Pierce County Homelessness Policy documents and 
converts them into a vector store.

https://www.piercecountywa.gov/7587/Homeless-Program-Policies

"""

import os
from VectorStore import VectorStore

##############################################################################
# Globals
##############################################################################
datafolder = '..\Data\\'
files = os.listdir(datafolder)

###############################################################################

vs = VectorStore()

docs = []
for f in files:

    if f[-3:] == 'pdf':
        print(r'################################################################')
        print(f'{f}')
        
        path = datafolder + f
        tempdoc = vs.loadDocuments(path)
        tempdoc = vs.removeIntro(tempdoc)
        tempdoc = vs.normalizeDocuments(tempdoc)
        splitdocs = vs.splitDocuments(documents=tempdoc)
        docs += splitdocs


vector_store = vs.buildVectorStore(docs)

vector_store.save_local(r"..\Vector_DB\faiss_index_homeless_policy")

