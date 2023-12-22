import cohere
import faiss
import numpy as np
import pandas as pd
import numpy as np

from llama_index import Document
from llama_index.node_parser import MarkdownNodeParser

reports_df = pd.read_csv('reports_final.csv')
report_content_ls = reports_df['content'].tolist()

markdown_docs = reports_df['content'].tolist()
source_ls = reports_df['source'].tolist()
documents = [Document(text=t, metadata={"df_index": s},) for t,s in zip(markdown_docs,list(range(len(markdown_docs))))]
parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(documents)
parsed_docs = [node.text for node in nodes]
parsed_docs_df_idx = [node.metadata["df_index"] for node in nodes]
print(f"Number of nodes: {len(nodes)}")



model_name = "embed-english-v3.0"
api_key = ""
input_type_embed = "search_document"

# Now we'll set up the cohere client.
co = cohere.Client(api_key)


# # Get the embeddings
# print("Getting embeddings...")
# embeds = co.embed(texts=parsed_docs,
#                   model=model_name,
#                   input_type=input_type_embed).embeddings
# # save the embeddings as a numpy array
# np.save('reports_embeds.npy', embeds)

# Load the embeddings
embeds = np.load('reports_embeds.npy')
embeds = embeds.astype(np.float32)
print(len(embeds))  


vector_dimension = embeds.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
embeds = np.ascontiguousarray(embeds)
faiss.normalize_L2(embeds)
index.add(embeds)


search_text = 'training using pytorch lightning'
search_text = 'fine tune llama 2 longlora'

# search_vector = encoder.encode(search_text)
search_vector = co.embed(texts=[search_text],
                  model=model_name,
                  input_type=input_type_embed).embeddings

# print(f"shape of search vector: {search_vector.shape}")
_vector = np.array([search_vector])
_vector = _vector.astype(np.float32)
_vector = np.ascontiguousarray(_vector)
# faiss.normalize_L2(_vector)
faiss.normalize_L2(_vector)


# reduce from shape (1, 21, 1024) to (21, 1024)
_vector = np.squeeze(_vector, axis=0)

k = index.ntotal
k = 50
distances, ann = index.search(_vector, k=k)

results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
parsed_docs_df = pd.DataFrame({'parsed_docs': parsed_docs, 'parsed_docs_df_idx': parsed_docs_df_idx})
# print(results.head(10))
# print(reports_df.columns)

merge = pd.merge(results, parsed_docs_df, left_on='ann', right_index=True)
# print(merge.head(10))
# print(merge[["display_name", "description"]].head(10))

# reports_df['display_name'][ii]

### Cohere re-ranker
# print(nodes[:5])

##### Cohere re-ranker
results = co.rerank(query=search_text, 
                    documents=merge["parsed_docs"], 
                    top_n=10,
                    model='rerank-english-v2.0') 

for idx, r in enumerate(results):
    print(f"Document Rank: {idx + 1}, Document Index: {r.index}")

    df_idx = merge["parsed_docs_df_idx"][r.index]

    print(f"Document Title: {reports_df['display_name'][df_idx]}")
    print(f"Document chunk: {merge['parsed_docs'][r.index][:200] + '...'}")
    #   print(f"Document: {r.document['text']}")
    print(f"Relevance Score: {r.relevance_score:.2f}")
    print("\n")