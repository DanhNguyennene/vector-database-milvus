import numpy as np
import pandas as pd
import json
from contextlib import contextmanager
import time
from pymilvus import Collection, connections

@contextmanager
def milvus_connection(alias="default", host='localhost', port='19530', retries=3, retry_delay=5):
    """Context manager for Milvus connection with retry logic"""
    for attempt in range(retries):
        try:
            connections.connect(alias=alias, host=host, port=port, grpc_max_message_length=200 * 1024 * 1024)
            yield
        except Exception as e:
            if attempt == retries - 1:
                raise Exception(f"Failed to connect after {retries} attempts: {str(e)}")
            print(f"Connection attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        finally:
            try:
                connections.disconnect(alias=alias)
            except:
                pass

def perform_search(collection, question_embedding, corpus_df, top_k=50, batch_size=100):
    """
    Perform search on Milvus collection with batching
    """
    search_params = {
        "metric_type": "COSINE",
        "params": {
            "M": 16,
            "efConstruction": 200
        }
    }
    
    # Process embeddings in batches
    qid_cid = []
    for i in range(0, len(question_embedding), batch_size):
        batch_embeddings = question_embedding[i:i+batch_size]
        
        results = collection.search(
            data=batch_embeddings,
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["id"]
        )
        
        for j, result in enumerate(results):
            # Correctly handle list comprehension
            list_index = [(hit.id, hit.distance) for hit in result]
            
            # Safely get chunk IDs and context IDs
            chunk_ids = [id for id, _ in list_index]
            corpus_cid = list(set(corpus_df[corpus_df["chunk_id"].isin(chunk_ids)]["cid"].tolist()))
            
            dictionary = {
                "qid": i + j,
                "cid": corpus_cid,
                "cosine": [dist for _, dist in list_index]
            }
            qid_cid.append(dictionary)
    
    return qid_cid

def main():
    # Configuration
    collection_name = "SOICT"
    host = 'localhost'
    port = '19530'
    output_file = "./search_results.json"
    
    try:
        # Load question embeddings and corpus data
        print("Loading data...")
        question_embedding = np.load("./data/question_embeddings_lite.npy")
        corpus_df = pd.read_csv("./data/chunked_corpus_lite.csv")
        
        # Connect to Milvus
        with milvus_connection(host=host, port=port):
            print("Connected to Milvus successfully")
            
            # Get collection
            collection = Collection(collection_name)
            
            # Load collection into memory if not already loaded
            print("Loading collection...")
            collection.load()
            
            print(f"Performing search for {len(question_embedding)} queries...")
            qid_cid = perform_search(collection, question_embedding, corpus_df)
            
            # Save results
            print(f"Saving results to {output_file}...")
            with open(output_file, 'w') as f:
                json.dump(qid_cid, f, indent=4)
            
            print(f"\nTotal queries processed: {len(qid_cid)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
