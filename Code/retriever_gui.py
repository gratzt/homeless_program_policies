# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:52:37 2024

@author: Trevor Gratz, trevormgratz@gmail.com

The following code creates a user interface to run semantic search on Pierce 
County's homeless program policies. The code may be adapted to allow for 
RAG, however, access to perfomant LLMs may preclude this option.  

https://www.piercecountywa.gov/7587/Homeless-Program-Policies

"""

import tkinter as tk
from rag_chain import RAG_Chain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

###############################################################################
# Load Embedding Model and Vector Store
###############################################################################

embed_mod = 'BAAI/bge-small-en-v1.5'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
    
embeddings = HuggingFaceBgeEmbeddings(
        model_name=embed_mod,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

vector_store = FAISS.load_local(
    r"..\Vector_DB\faiss_index_homeless_policy", embeddings, allow_dangerous_deserialization=True
)
rc = RAG_Chain()
qa_chain = rc.createRAGChain()

###############################################################################
# Build GUI
###############################################################################

# Function to handle the search logic
def search():
    user_input = entry_input.get()  # Get user input from the entry field
    # Retreiveal
    output = qa_chain.invoke(user_input).text
    result = output  # Result from the model

    # Insert the result in the scrollable text widget
    text_result.delete(1.0, tk.END)  # Clear previous content
    text_result.insert(tk.END, result)  # Insert new result

# Function to close the application
def close_application():
    root.quit()

# Create the main window
root = tk.Tk()
root.title("String Search GUI")

# Set the window size
root.geometry("800x1000")

# Create a label for the user input
label_input = tk.Label(root, text="Enter your question:")
label_input.pack(pady=10)

# Create an entry widget for the user input
entry_input = tk.Entry(root, width=40)
entry_input.pack(pady=5)

# Create a search button
button_search = tk.Button(root, text="Search", command=search)
button_search.pack(pady=10)

# Create a frame for the Text widget and the Scrollbar
frame_result = tk.Frame(root)
frame_result.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Create a Text widget for displaying the result with wrapping and scrolling
text_result = tk.Text(frame_result, wrap=tk.WORD, height=15, width=80)
text_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a Scrollbar for the Text widget
scrollbar = tk.Scrollbar(frame_result, command=text_result.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Link the scrollbar to the Text widget
text_result.config(yscrollcommand=scrollbar.set)

# Create a close button
button_close = tk.Button(root, text="Close", command=close_application)
button_close.pack(pady=10)

# Run the main loop of the GUI
root.mainloop()
