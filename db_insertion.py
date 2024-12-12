from pymilvus import (
    FieldSchema, 
    CollectionSchema, 
    DataType, 
    Collection, 
    connections, 
    utility
)
import csv
import numpy as np
import pandas as pd
import json
from multiprocessing import Pool, cpu_count
import time
from contextlib import contextmanager

# Connection management
@contextmanager
def milvus_connection(alias, host='localhost', port='19530', retries=3, retry_delay=5):
    """Context manager for Milvus connection with retry logic"""
    for attempt in range(retries):
        try:
            connections.connect(alias=alias, host=host, port=port)
            yield
            break
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

def load_data(file_path):
    """Load IDs from CSV file"""
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            return [int(row[2]) for row in reader]
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {str(e)}")

def insert_batch(args):
    """Insert a batch of data into Milvus"""
    name, start_idx, end_idx, batch_ids, batch_embeddings = args
    connection_alias = f"worker_{start_idx}"
    
    try:
        with milvus_connection(alias=connection_alias):
            collection = Collection(name=name)
            collection.insert([batch_ids, batch_embeddings])
            print(f"Successfully inserted batch {start_idx}:{end_idx}")
            return True, None
    except Exception as e:
        error_msg = f"Error inserting batch {start_idx}:{end_idx}: {str(e)}"
        print(error_msg)
        return False, error_msg

def create_collection(name, dim=768):
    """Create a new collection with proper error handling"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Collection for Questions and Corpus")
    return Collection(name=name, schema=schema)

def main():
    name = "SOICT"
    host = 'localhost'
    port = '19530'
    
    try:
        print("Loading data...")
        corpus_embedding = np.load("./data/corpus_lite.npy")
        chunk_id = load_data("./data/chunked_corpus_lite.csv")
        
        # Main connection for setup
        with milvus_connection("default", host=host, port=port):
            # Create or get collection
            if not utility.has_collection(name):
                print(f"Creating new collection '{name}'...")
                collection = create_collection(name)
            else:
                print(f"Collection '{name}' exists, recreating for fresh start...")
                utility.drop_collection(name)
                collection = create_collection(name)
            
            # Configure parallel insertion
            batch_size = 1000
            n_processes = min(4, cpu_count())
            total_size = len(chunk_id)
            
            # Prepare batches
            batches = []
            for start_idx in range(0, total_size, batch_size):
                end_idx = min(start_idx + batch_size, total_size)
                batch_ids = chunk_id[start_idx:end_idx].copy()
                batch_embeddings = corpus_embedding[start_idx:end_idx].copy()
                batches.append((name, start_idx, end_idx, batch_ids, batch_embeddings))
            
            # Parallel insertion
            print(f"Starting parallel insertion with {n_processes} processes...")
            with Pool(processes=n_processes) as pool:
                results = pool.map(insert_batch, batches)
            
            # Check results
            success_count = sum(1 for success, _ in results if success)
            print(f"Inserted {success_count}/{len(batches)} batches successfully")
            
            if success_count < len(batches):
                print("Failed batches:")
                for success, error in results:
                    if not success:
                        print(error)
                raise Exception("Some batches failed to insert")
            
            # Create index
            print("Creating index...")
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 768}
            }
            
            if collection.has_index():
                print("Dropping existing index...")
                collection.release()
                collection.drop_index()
            
            collection.create_index(field_name="vector", index_params=index_params)
            print("Index created successfully!")
            
            # Load collection
            collection.load()
            print("Collection loaded successfully!")
            
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
