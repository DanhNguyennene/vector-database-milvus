from pymilvus import Collection, connections, utility
import time
from contextlib import contextmanager

@contextmanager
def milvus_connection(alias="default", host='localhost', port='19530', retries=3, retry_delay=5):
    """Context manager for Milvus connection with retry logic"""
    for attempt in range(retries):
        try:
            connections.connect(alias=alias, host=host, port=port)
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

def replace_index(collection_name, index_params=None):
    """
    Replace the index of a collection with new parameters
    """
    if index_params is None:
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 768}
        }
    
    try:
        collection = Collection(collection_name)
        
        # Check if collection is loaded
        if collection.is_loaded():
            print("Releasing collection from memory...")
            collection.release()
        
        # Drop existing index if it exists
        if collection.has_index():
            print("Dropping existing index...")
            collection.drop_index()
            print("Existing index dropped successfully")
        
        # Create new index
        print(f"Creating new index with parameters: {index_params}")
        collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        print("New index created successfully")
        
        # Load collection back into memory
        print("Loading collection into memory...")
        collection.load()
        print("Collection loaded successfully")
        
        return True
    
    except Exception as e:
        print(f"Error during index replacement: {str(e)}")
        return False

def main():
    # Configuration
    collection_name = "SOICT"
    host = 'localhost'
    port = '19530'
    
    # You can customize the index parameters here
    custom_index_params = {
        "index_type": "IVF_FLAT",  # Options: IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, etc.
        "metric_type": "COSINE",   # Options: L2, IP, COSINE, etc.
        "params": {
            "nlist": 768           # Number of clusters
        }
    }
    
    try:
        print(f"Connecting to Milvus server at {host}:{port}")
        with milvus_connection(host=host, port=port):
            # Check if collection exists
            if not utility.has_collection(collection_name):
                raise Exception(f"Collection '{collection_name}' does not exist")
            
            print(f"Starting index replacement for collection '{collection_name}'")
            start_time = time.time()
            
            # Replace index
            success = replace_index(collection_name, custom_index_params)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if success:
                print(f"\nIndex replacement completed successfully")
                print(f"Time taken: {duration:.2f} seconds")
            else:
                print("\nIndex replacement failed")
            
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
