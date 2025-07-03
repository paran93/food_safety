import pandas as pd
import numpy as np
import pickle
import os
import hashlib
import faiss
from sentence_transformers import SentenceTransformer
import warnings
import sys
from typing import List, Dict, Tuple, Any # Import List, Dict, Tuple, Any

warnings.filterwarnings('ignore')

# Set up a dummy Streamlit-like info/progress for local execution
# This allows the NYCFoodSafetyRAG class to run without a Streamlit environment
class DummyStreamlit:
    def info(self, message):
        print(f"INFO: {message}")
    def success(self, message):
        print(f"SUCCESS: {message}")
    def warning(self, message):
        print(f"WARNING: {message}")
    def error(self, message):
        print(f"ERROR: {message}")
    def progress(self, value, text=""):
        # Simple print for progress, not a real progress bar
        sys.stdout.write(f"\rPROGRESS: {text} {int(value*100)}%")
        sys.stdout.flush()
    def write(self, *args, **kwargs):
        # Convert all args to string and print
        print("DEBUG:", ' '.join(map(str, args)))

# Replace Streamlit functions with dummy ones for local execution
st = DummyStreamlit()

# --- Re-use your NYCFoodSafetyRAG class definition ---
# This class is designed to be runnable both in Streamlit and standalone
# for asset generation.
class NYCFoodSafetyRAG:
    def __init__(self):
        self.df = None
        self.model = None
        self.cache_dir = "cache" # Ensure this cache directory exists locally
        self.setup_cache()
        self.index = None
        self.violation_embeddings = None
        self.restaurant_embeddings = None
        self.processed_violations = []
        self.processed_restaurants = []

    def get_cache_key(self, identifier, sample_size=None):
        """Generate unique cache key based on identifier (file path or URL) and parameters"""
        content = f"{identifier}_{sample_size}"
        # If identifier is a file path, also include file modification time and size
        if os.path.exists(identifier) and not identifier.startswith(('http://', 'https://')):
            try:
                file_stat = os.stat(identifier)
                content += f"_{file_stat.st_mtime}_{file_stat.st_size}"
            except Exception as e:
                st.warning(f"Could not get file stats for cache key: {e}")
        return hashlib.md5(content.encode()).hexdigest().encode('utf-8')

    # No @st.cache_data here because we're doing manual caching/saving in this script
    def load_data(self, data_source_identifier, sample_size=None):
        st.info("Loading NYC Food Safety dataset...")

        # No caching logic here, load fresh data for asset generation
        if data_source_identifier.startswith(('http://', 'https://')):
            st.error("This script expects a local file path for direct CSV reading during asset generation.")
            st.error("Please ensure your full NY_food_safety.csv is locally accessible.")
            sys.exit(1) # Exit if it's a URL in generation script
        else:
            csv_content_source = data_source_identifier

        try:
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            temp_df = None
            for encoding in encodings:
                try:
                    if sample_size:
                        chunk_iterator = pd.read_csv(csv_content_source, chunksize=10000, encoding=encoding)
                        chunks = []
                        total_rows = 0
                        for chunk in chunk_iterator:
                            chunks.append(chunk)
                            total_rows += len(chunk)
                            if total_rows >= sample_size:
                                break
                        temp_df = pd.concat(chunks, ignore_index=True).head(sample_size)
                    else:
                        temp_df = pd.read_csv(csv_content_source, encoding=encoding)

                    if 'SCORE' in temp_df.columns:
                        self.df = temp_df
                        st.info(f"Dataset loaded using encoding '{encoding}': {len(self.df)} records")
                        break
                    else:
                        st.warning(f"Encoding '{encoding}' loaded data, but 'SCORE' column not found. Columns: {temp_df.columns.tolist()}")

                except Exception as inner_e:
                    st.info(f"Failed to load with encoding '{encoding}': {inner_e}")
            
            if self.df is None:
                raise ValueError("Could not load data or 'SCORE' column not found with any tested encoding.")

            st.write("Columns after loading (before preprocessing):", self.df.columns.tolist())

        except Exception as e:
            st.error(f"❌ Critical Error: Failed to load data from {data_source_identifier}: {e}")
            sys.exit(1)

        self._preprocess_data()
        return self.df

    def setup_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _preprocess_data(self):
        st.info("Preprocessing data...")
        self.df.columns = self.df.columns.str.strip().str.upper()

        st.write("Columns after normalization (before type conversion):", self.df.columns.tolist())

        if 'SCORE' not in self.df.columns:
            st.error("Error: 'SCORE' column not found after normalizing column names. Available columns: " + str(self.df.columns.tolist()))
            sys.exit(1)

        date_columns = ['INSPECTION DATE', 'GRADE DATE', 'RECORD DATE']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            else:
                st.warning(f"Date column '{col}' not found in data.")


        text_columns = ['DBA', 'VIOLATION DESCRIPTION', 'CUISINE DESCRIPTION', 'BORO', 'CRITICAL FLAG', 'GRADE']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('').astype(str)
            else:
                st.warning(f"Text column '{col}' not found in data.")

        self.df['SCORE'] = pd.to_numeric(self.df['SCORE'], errors='coerce').fillna(0)
        self.df['GRADE'] = self.df['GRADE'].replace('', 'Not Graded').fillna('Not Graded')
        self.df['CRITICAL FLAG'] = self.df['CRITICAL FLAG'].replace('', 'Not Critical').fillna('Not Critical')
        self.df['BORO'] = self.df['BORO'].replace('', 'Unknown').fillna('Unknown')
        self.df['CUISINE DESCRIPTION'] = self.df['CUISINE DESCRIPTION'].replace('', 'Unknown').fillna('Unknown')


        if 'INSPECTION DATE' in self.df.columns and not self.df['INSPECTION DATE'].isnull().all():
            self.df['YEAR'] = self.df['INSPECTION DATE'].dt.year
            self.df['MONTH'] = self.df['INSPECTION DATE'].dt.month
        else:
            st.warning("INSPECTION DATE column is missing or entirely null, cannot create YEAR/MONTH features.")
            self.df['YEAR'] = 0
            self.df['MONTH'] = 0

        self.df['IS_CRITICAL'] = (self.df['CRITICAL FLAG'] == 'Critical').astype(int)
        st.success("Data preprocessing completed!")

    # No @st.cache_resource here, as we're generating assets
    def build_rag_system(self, max_docs=None):
        st.info("Building optimized RAG system...")

        # For generation, always rebuild
        st.info("🔄 Building embeddings from scratch...")
        self._prepare_documents_optimized(max_docs)
        self._create_embeddings_optimized()

        # Build and cache search index
        self._build_search_index()
        st.success("🚀 RAG system ready for saving!")


    def _prepare_documents_optimized(self, max_docs=None):
        st.info("Preparing documents for embedding...")
        df_sample = self.df.head(max_docs) if max_docs else self.df
        
        violation_docs = []
        for _, row in df_sample.iterrows():
            doc = {
                'id': f"violation_{len(violation_docs)}",
                'restaurant_name': row.get('DBA', ''),
                'borough': row.get('BORO', ''),
                'cuisine': row.get('CUISINE DESCRIPTION', ''),
                'violation_code': row.get('VIOLATION CODE', ''),
                'violation_description': row.get('VIOLATION DESCRIPTION', ''),
                'critical_flag': row.get('CRITICAL FLAG', ''),
                'score': row.get('SCORE', 0),
                'grade': row.get('GRADE', 'Not Graded'),
                'inspection_date': str(row.get('INSPECTION DATE', pd.NaT)),
                'inspection_type': row.get('INSPECTION TYPE', ''),
                'text': f"Restaurant: {row.get('DBA', 'N/A')} in {row.get('BORO', 'N/A')} serving {row.get('CUISINE DESCRIPTION', 'N/A')} cuisine. "
                       f"Violation: {row.get('VIOLATION DESCRIPTION', 'N/A')} (Code: {row.get('VIOLATION CODE', 'N/A')}). "
                       f"Critical: {row.get('CRITICAL FLAG', 'N/A')}. Score: {row.get('SCORE', 'N/A')}. Grade: {row.get('GRADE', 'N/A')}."
            }
            violation_docs.append(doc)
        self.processed_violations = violation_docs

        restaurant_summaries = []
        restaurant_groups = df_sample.groupby('CAMIS')

        for camis, group in restaurant_groups:
            restaurant_info = group.iloc[0]
            violations_list = group['VIOLATION DESCRIPTION'].tolist()
            critical_count = len(group[group['CRITICAL FLAG'] == 'Critical'])
            avg_score = group['SCORE'].mean()
            recent_grade = group['GRADE'].mode()[0] if not group['GRADE'].mode().empty else 'N/A'

            doc = {
                'id': f"restaurant_{camis}",
                'camis': camis,
                'name': restaurant_info.get('DBA', ''),
                'borough': restaurant_info.get('BORO', ''),
                'cuisine': restaurant_info.get('CUISINE DESCRIPTION', ''),
                'total_violations': len(violations_list),
                'critical_violations': critical_count,
                'avg_score': avg_score,
                'recent_grade': recent_grade,
                'text': f"Restaurant: {restaurant_info.get('DBA', 'N/A')} in {restaurant_info.get('BORO', 'N/A')} "
                       f"serving {restaurant_info.get('CUISINE DESCRIPTION', 'N/A')} cuisine. "
                       f"Total violations: {len(violations_list)}, Critical violations: {critical_count}. "
                       f"Average score: {avg_score:.1f}, Recent grade: {recent_grade}. "
                       f"Common violations: {', '.join(violations_list[:3]) if violations_list else 'None reported'}"
            }
            restaurant_summaries.append(doc)
        self.processed_restaurants = restaurant_summaries
        st.info(f"Prepared {len(self.processed_violations)} violation documents and {len(self.processed_restaurants)} restaurant summaries")


    def _create_embeddings_optimized(self):
        st.info("Creating embeddings...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # Load model locally for generation

        violation_texts = [doc['text'] for doc in self.processed_violations]
        # Simplified progress update for local script
        batch_size = 32
        self.violation_embeddings = []
        for i in range(0, len(violation_texts), batch_size):
            batch_texts = violation_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size)
            self.violation_embeddings.extend(batch_embeddings)
            st.progress(min(1.0, (i + batch_size) / len(violation_texts)), text=f"Creating violation embeddings... {min(100, int((i + batch_size) / len(violation_texts) * 100))}%")
        self.violation_embeddings = np.array(self.violation_embeddings)
        st.progress(1.0, text="Violation embeddings created!")


        restaurant_texts = [doc['text'] for doc in self.processed_restaurants]
        # Simplified progress update for local script
        self.restaurant_embeddings = []
        for i in range(0, len(restaurant_texts), batch_size):
            batch_texts = restaurant_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size)
            self.restaurant_embeddings.extend(batch_embeddings)
            st.progress(min(1.0, (i + batch_size) / len(restaurant_texts)), text=f"Creating restaurant embeddings... {min(100, int((i + batch_size) / len(restaurant_texts) * 100))}%")
        self.restaurant_embeddings = np.array(self.restaurant_embeddings)
        st.progress(1.0, text="Restaurant embeddings created!")

        st.info(f"Created embeddings: {self.violation_embeddings.shape[0]} violations, {self.restaurant_embeddings.shape[0]} restaurants")

    def _build_search_index(self):
        st.info("Building search index...")
        all_embeddings = np.vstack([self.violation_embeddings, self.restaurant_embeddings])
        dimension = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(all_embeddings)
        self.index.add(all_embeddings.astype('float32'))
        st.info(f"Search index built with {self.index.ntotal} documents")

    def search(self, query: str, top_k: int = 10, search_type: str = 'all') -> List[Dict]:
        if self.model is None or self.index is None:
            st.error("RAG system not built. Call build_rag_system() first.")
            return []

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)

        results = []
        violation_count = len(self.processed_violations)

        for score, idx in zip(scores[0], indices[0]):
            if idx < violation_count:
                if search_type in ['all', 'violations']:
                    doc = self.processed_violations[idx].copy()
                    doc['score'] = float(score)
                    doc['type'] = 'violation'
                    results.append(doc)
            else:
                if search_type in ['all', 'restaurants']:
                    doc = self.processed_restaurants[idx - violation_count].copy()
                    doc['score'] = float(score)
                    doc['type'] = 'restaurant'
                    results.append(doc)

        return results[:top_k]

# --- Main execution block for generating assets ---
if __name__ == "__main__":
    # Path to your full local CSV file (the 20MB version)
    DATA_SOURCE_PATH = 'NY_food_safety.csv'
    # Set this to None to process the entire dataset, or an integer for a sample
    MAX_DOCS_FOR_EMBEDDING = None # Use None for full dataset, or e.g., 50000 for a large sample

    rag_system_generator = NYCFoodSafetyRAG()

    print("\n--- Starting Asset Generation ---")
    print(f"Loading data from: {DATA_SOURCE_PATH}")
    print(f"Generating embeddings for up to {MAX_DOCS_FOR_EMBEDDING if MAX_DOCS_FOR_EMBEDDING else 'all'} documents.")

    # Load data
    rag_system_generator.load_data(DATA_SOURCE_PATH, sample_size=MAX_DOCS_FOR_EMBEDDING)

    # Build RAG system (this will generate embeddings)
    rag_system_generator.build_rag_system(max_docs=MAX_DOCS_FOR_EMBEDDING)

    # Define paths for saving assets
    assets_dir = "precomputed_assets"
    os.makedirs(assets_dir, exist_ok=True)

    violations_pkl_path = os.path.join(assets_dir, "processed_violations.pkl")
    restaurants_pkl_path = os.path.join(assets_dir, "processed_restaurants.pkl")
    violation_embeddings_npy_path = os.path.join(assets_dir, "violation_embeddings.npy")
    restaurant_embeddings_npy_path = os.path.join(assets_dir, "restaurant_embeddings.npy")
    faiss_index_path = os.path.join(assets_dir, "faiss_index.bin")

    print(f"\n--- Saving Precomputed Assets to '{assets_dir}' ---")

    with open(violations_pkl_path, 'wb') as f:
        pickle.dump(rag_system_generator.processed_violations, f)
    with open(restaurants_pkl_path, 'wb') as f:
        pickle.dump(rag_system_generator.processed_restaurants, f)
    np.save(violation_embeddings_npy_path, rag_system_generator.violation_embeddings)
    np.save(restaurant_embeddings_npy_path, rag_system_generator.restaurant_embeddings)
    faiss.write_index(rag_system_generator.index, faiss_index_path)