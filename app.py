import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import hashlib
import faiss
from sentence_transformers import SentenceTransformer
import warnings
import matplotlib.pyplot as plt # Kept for completeness, though Plotly is primary
import seaborn as sns # Kept for completeness
import plotly.express as px
import requests # Kept for general web requests if needed, not for CSV load
import io
from typing import List, Dict, Tuple, Any

warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

class NYCFoodSafetyRAG:
    def __init__(self):
        self.df = None
        self.model = None
        self.cache_dir = "cache" # Directory for Streamlit's internal cache
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

    @st.cache_data(show_spinner="⏳ Loading and preprocessing data...")
    def load_data(_self, data_source_identifier, sample_size=None): # Changed 'self' to '_self'
        """
        Load and preprocess data with caching.
        data_source_identifier is expected to be a local file path for this app.
        """
        st.info("Loading NYC Food Safety dataset...")

        cache_key = _self.get_cache_key(data_source_identifier, sample_size)
        cache_file = os.path.join(_self.cache_dir, f"data_{cache_key.decode('utf-8')}.pkl")

        if os.path.exists(cache_file):
            st.info("📦 Loading preprocessed data from cache...")
            try:
                with open(cache_file, 'rb') as f:
                    _self.df = pickle.load(f)
                st.success(f"✅ Cached data loaded: {len(_self.df):,} records")
                return _self.df
            except Exception as e:
                st.warning(f"⚠️ Data cache corrupted, reloading: {e}")
                os.remove(cache_file)

        # Expecting a local file path from GitHub
        if data_source_identifier.startswith(('http://', 'https://')):
            st.error("This app is configured to load CSV from a local GitHub path. "
                     "Please ensure 'NY_food_safety.csv' is in your GitHub repo root.")
            st.stop()
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
                        _self.df = temp_df
                        st.info(f"Dataset loaded using encoding '{encoding}': {len(_self.df)} records")
                        break
                    else:
                        st.warning(f"Encoding '{encoding}' loaded data, but 'SCORE' column not found. Columns: {temp_df.columns.tolist()}")

                except Exception as inner_e:
                    st.info(f"Failed to load with encoding '{encoding}': {inner_e}")
            
            if _self.df is None:
                raise ValueError("Could not load data or 'SCORE' column not found with any tested encoding.")

            st.write("Columns after loading (before preprocessing):", _self.df.columns.tolist())

        except Exception as e:
            st.error(f"❌ Critical Error: Failed to load data from {data_source_identifier}: {e}")
            st.exception(e)
            st.stop()

        _self._preprocess_data()

        st.info("💾 Caching preprocessed data...")
        with open(cache_file, 'wb') as f:
            pickle.dump(_self.df, f)
        st.success("✅ Data cached for future runs")

        return _self.df

    def setup_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _preprocess_data(self):
        st.info("Preprocessing data...")
        self.df.columns = self.df.columns.str.strip().str.upper()

        st.write("Columns after normalization (before type conversion):", self.df.columns.tolist())

        if 'SCORE' not in self.df.columns:
            st.error("Error: 'SCORE' column not found after normalizing column names. Available columns: " + str(self.df.columns.tolist()))
            st.stop()

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

    @st.cache_resource(show_spinner="⚙️ Building RAG system...")
    def build_rag_system(_self, max_docs=None): # Changed 'self' to '_self'
        st.info("Building optimized RAG system...")

        # Define paths for precomputed assets
        assets_dir = "precomputed_assets"
        violations_pkl_path = os.path.join(assets_dir, "processed_violations.pkl")
        restaurants_pkl_path = os.path.join(assets_dir, "processed_restaurants.pkl")
        violation_embeddings_npy_path = os.path.join(assets_dir, "violation_embeddings.npy")
        restaurant_embeddings_npy_path = os.path.join(assets_dir, "restaurant_embeddings.npy")
        faiss_index_path = os.path.join(assets_dir, "faiss_index.bin")
        
        # Check if precomputed assets exist
        if (os.path.exists(violations_pkl_path) and
            os.path.exists(restaurants_pkl_path) and
            os.path.exists(violation_embeddings_npy_path) and
            os.path.exists(restaurant_embeddings_npy_path) and
            os.path.exists(faiss_index_path)):

            st.info("📦 Loading precomputed embeddings and FAISS index...")
            try:
                with open(violations_pkl_path, 'rb') as f:
                    _self.processed_violations = pickle.load(f)
                with open(restaurants_pkl_path, 'rb') as f:
                    _self.processed_restaurants = pickle.load(f)
                _self.violation_embeddings = np.load(violation_embeddings_npy_path)
                _self.restaurant_embeddings = np.load(restaurant_embeddings_npy_path)
                
                _self.index = faiss.read_index(faiss_index_path)
                
                # Load the SentenceTransformer model (it's small and fast to load from cache)
                model_cache_path = os.path.join(_self.cache_dir, "sentence_transformer_model")
                if os.path.exists(model_cache_path):
                    _self.model = SentenceTransformer(model_cache_path)
                else: # Download if not cached, though it should be if load_data ran.
                    _self.model = SentenceTransformer('all-MiniLM-L6-v2')
                    _self.model.save(model_cache_path)

                st.success("✅ Precomputed embeddings and FAISS index loaded!")
            except Exception as e:
                st.warning(f"⚠️ Failed to load precomputed assets, rebuilding from scratch: {e}")
                # Clean up potentially corrupted files
                for p in [violations_pkl_path, restaurants_pkl_path, violation_embeddings_npy_path, restaurant_embeddings_npy_path, faiss_index_path]:
                    if os.path.exists(p): os.remove(p)
                # Fall through to rebuild logic
                _self._rebuild_assets_from_scratch(max_docs)
        else:
            st.info("🔄 Precomputed assets not found or incomplete, building from scratch...")
            _self._rebuild_assets_from_scratch(max_docs)

        st.success("🚀 RAG system ready!")

    def _rebuild_assets_from_scratch(_self, max_docs=None):
        """Helper to encapsulate the embedding generation process"""
        st.info("Loading sentence transformer model...")
        _self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Save model locally for future reloads (Streamlit cache takes care of this too)
        _self.model.save(os.path.join(_self.cache_dir, "sentence_transformer_model"))

        _self._prepare_documents_optimized(max_docs)
        _self._create_embeddings_optimized()
        _self._build_search_index()


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
        # Note: Model loading happens in _rebuild_assets_from_scratch
        
        violation_texts = [doc['text'] for doc in self.processed_violations]
        progress_violation = st.progress(0, text="Creating violation embeddings...")
        self.violation_embeddings = []
        batch_size = 32
        for i in range(0, len(violation_texts), batch_size):
            batch_texts = violation_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size)
            self.violation_embeddings.extend(batch_embeddings)
            progress_violation.progress(min(1.0, (i + batch_size) / len(violation_texts)), text=f"Creating violation embeddings... {min(100, int((i + batch_size) / len(violation_texts) * 100))}%")
        self.violation_embeddings = np.array(self.violation_embeddings)
        progress_violation.progress(1.0, text="Violation embeddings created!")


        restaurant_texts = [doc['text'] for doc in self.processed_restaurants]
        progress_restaurant = st.progress(0, text="Creating restaurant embeddings...")
        self.restaurant_embeddings = []
        for i in range(0, len(restaurant_texts), batch_size):
            batch_texts = restaurant_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size)
            self.restaurant_embeddings.extend(batch_embeddings)
            progress_restaurant.progress(min(1.0, (i + batch_size) / len(restaurant_texts)), text=f"Creating restaurant embeddings... {min(100, int((i + batch_size) / len(restaurant_texts) * 100))}%")
        self.restaurant_embeddings = np.array(self.restaurant_embeddings)
        progress_restaurant.progress(1.0, text="Restaurant embeddings created!")

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


# =============================================================================
# INTELLIGENT ASSISTANT INTERFACE (Streamlit Friendly)
# =============================================================================

class FoodSafetyAssistant:
    def __init__(self, rag_system: NYCFoodSafetyRAG):
        self.rag = rag_system

    def _classify_query(self, question: str) -> str:
        """Classify the type of query"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['analyze', 'analysis', 'pattern', 'trend', 'statistics', 'chart', 'graph', 'distribution']):
            return 'analysis'
        elif any(word in question_lower for word in ['find', 'search', 'show me', 'list', 'details', 'info', 'record', 'violations for', 'inspection results']):
            return 'search'
        elif any(word in question_lower for word in ['recommend', 'suggest', 'best', 'avoid', 'safest', 'worst']):
            return 'recommendation'
        else:
            return 'general'

    def ask(self, question: str) -> Tuple[str, Any]: # Return a tuple of (response_text, plot_object)
        """Ask the intelligent assistant a question"""
        query_type = self._classify_query(question)

        if query_type == 'analysis':
            return self._handle_analysis_query(question)
        elif query_type == 'search':
            return self._handle_search_query(question) # This now returns (text, map_object)
        elif query_type == 'recommendation':
            return self._handle_recommendation_query(question), None
        else:
            return self._handle_general_query(question), None

    def _handle_analysis_query(self, question: str) -> Tuple[str, Any]:
        """Handle analysis-type queries, returning text and a Plotly figure"""
        response_text = ""
        plot_object = None

        if 'critical' in question.lower() and 'violations' in question.lower():
            critical_df = self.rag.df[self.rag.df['CRITICAL FLAG'] == 'Critical']
            total_critical = len(critical_df)
            critical_rate = total_critical / len(self.rag.df) * 100

            # Get top 10 for better plot
            top_critical = critical_df['VIOLATION DESCRIPTION'].value_counts().head(10)

            response_text += f"📊 **Critical Violations Analysis:**\n\n"
            response_text += f"• Total critical violations: {total_critical:,} ({critical_rate:.1f}% of all violations)\n\n"
            response_text += f"**Top 10 Critical Violations:**\n"
            for i, (violation, count) in enumerate(top_critical.items(), 1):
                response_text += f"{i}. {violation[:80]}... ({count:,} cases)\n"

            # Create a Plotly bar chart for critical violations
            fig_critical = px.bar(
                top_critical,
                x=top_critical.values,
                y=top_critical.index,
                orientation='h',
                title='Top 10 Critical Violations',
                labels={'x': 'Number of Occurrences', 'y': 'Violation Description'},
                color_discrete_sequence=px.colors.qualitative.Pastel # Add a nice color palette
            )
            fig_critical.update_layout(yaxis={'autorange': "reversed"}) # show highest bar on top
            plot_object = fig_critical

        elif 'borough' in question.lower():
            borough_stats = self.rag.df.groupby('BORO').agg(
                Unique_Restaurants=('CAMIS', 'nunique'),
                Avg_Score=('SCORE', 'mean'),
                Critical_Rate=('IS_CRITICAL', 'mean')
            ).round(2).reset_index() # Reset index to make BORO a column for Plotly

            response_text += f"🗺️ **Borough Analysis:**\n\n"
            for index, stats in borough_stats.iterrows():
                response_text += f"**{stats['BORO']}:**\n"
                response_text += f"  • Restaurants: {stats['Unique_Restaurants']:,}\n"
                response_text += f"  • Avg Score: {stats['Avg_Score']:.1f}\n"
                response_text += f"  • Critical Rate: {stats['Critical_Rate']*100:.1f}%\n\n"

            # Create a Plotly chart for borough critical rate
            fig_borough = px.bar(
                borough_stats,
                x='BORO',
                y='Critical_Rate',
                title='Critical Violation Rate by Borough',
                labels={'Critical_Rate': 'Critical Rate (%)', 'BORO': 'Borough'},
                color='Critical_Rate', # Color by critical rate
                color_continuous_scale=px.colors.sequential.YlOrRd # Use a red color scale for critical rates
            )
            fig_borough.update_layout(yaxis_tickformat=".1%") # Format y-axis as percentage
            plot_object = fig_borough

        elif 'grade' in question.lower() and 'distribution' in question.lower():
            grade_dist = self.rag.df['GRADE'].value_counts().reset_index()
            grade_dist.columns = ['Grade', 'Count']
            response_text += f"📈 **Grade Distribution:**\n\n"
            for index, row in grade_dist.iterrows():
                response_text += f"• Grade {row['Grade']}: {row['Count']:,} ({row['Count']/len(self.rag.df)*100:.1f}%)\n"

            fig_grade = px.pie(
                grade_dist,
                values='Count',
                names='Grade',
                title='Distribution of Restaurant Grades',
                hole=0.3, # Donut chart
                color_discrete_sequence=px.colors.qualitative.Pastel # Use a nice color palette
            )
            plot_object = fig_grade

        else:
            response_text = "I can provide analysis on critical violations, borough statistics, or grade distribution. Please specify."

        return response_text, plot_object

    def _handle_search_query(self, question: str) -> Tuple[str, Any]: # Modified return type to include Any for plot
        """Handle search-type queries with improved output for restaurants and a map."""
        results = self.rag.search(question, top_k=10) # Get more results to ensure we capture all relevant records for a restaurant

        if not results:
            return "❌ No relevant restaurants or violations found for your query. Try rephrasing or be more specific.", None

        response_parts = []
        map_data = [] # To store data for the map
        
        # --- Group results by restaurant CAMIS ---
        # Collect unique CAMIS from search results, and then pull full data from main DataFrame
        relevant_camis = set()
        for res in results:
            if res['type'] == 'restaurant':
                relevant_camis.add(res['camis'])
            elif res['type'] == 'violation':
                # Find CAMIS in the original DataFrame based on restaurant name (DBA)
                matching_rows = self.rag.df[self.rag.df['DBA'].str.contains(res['restaurant_name'], case=False, na=False)]
                if not matching_rows.empty:
                    relevant_camis.update(matching_rows['CAMIS'].unique())

        # Now, gather all latest info for these CAMIS from the main dataframe
        # Sort by inspection date to get the most recent valid location data and grade
        unique_restaurants_summary = self.rag.df[self.rag.df['CAMIS'].isin(relevant_camis)] \
                                                    .sort_values(by='INSPECTION DATE', ascending=False) \
                                                    .groupby('CAMIS') \
                                                    .first() \
                                                    .reset_index()

        # Build data for the map and the detailed text response
        restaurants_found_for_text = {} # Dictionary to build the structured text response

        for _, row in unique_restaurants_summary.iterrows():
            camis = row['CAMIS']
            
            # Filter all violations for this specific CAMIS from the original DataFrame
            restaurant_violations_df = self.rag.df[(self.rag.df['CAMIS'] == camis)].copy()
            
            # Prepare data for map
            # Ensure LATITUDE and LONGITUDE are available and not NaN
            if pd.notna(row.get('LATITUDE')) and pd.notna(row.get('LONGITUDE')):
                map_data.append({
                    'name': row.get('DBA', 'N/A'),
                    'latitude': row.get('LATITUDE'),
                    'longitude': row.get('LONGITUDE'),
                    'grade': row.get('GRADE', 'N/A'),
                    'score': row.get('SCORE', 0),
                    'critical_violations': len(restaurant_violations_df[restaurant_violations_df['CRITICAL FLAG'] == 'Critical']),
                    'cuisine': row.get('CUISINE DESCRIPTION', 'N/A'),
                    'borough': row.get('BORO', 'N/A')
                })
            
            # Prepare data for text response (similar to previous improved logic)
            data = {
                'name': row.get('DBA', 'N/A'),
                'borough': row.get('BORO', 'N/A'),
                'cuisine': row.get('CUISINE DESCRIPTION', 'N/A'),
                'total_violations': len(restaurant_violations_df), # Total inspections for this CAMIS
                'critical_violations': len(restaurant_violations_df[restaurant_violations_df['CRITICAL FLAG'] == 'Critical']),
                'avg_score': restaurant_violations_df['SCORE'].mean(),
                'recent_grade': row.get('GRADE', 'N/A'), # Use the most recent grade from groupby().first()
                'violations_detail': []
            }
            
            # Populate violations_detail for text response (top 5 most recent violations)
            for _, violation_row in restaurant_violations_df.sort_values(by='INSPECTION DATE', ascending=False).head(5).iterrows():
                data['violations_detail'].append({
                    'description': violation_row.get('VIOLATION DESCRIPTION', 'N/A'),
                    'critical': violation_row.get('CRITICAL FLAG', 'N/A'),
                    'score': violation_row.get('SCORE', 0),
                    'date': str(violation_row.get('INSPECTION DATE', pd.NaT).date()) if pd.notna(violation_row.get('INSPECTION DATE')) else 'N/A'
                })
            
            restaurants_found_for_text[camis] = data # Store for text processing


        if not restaurants_found_for_text:
             return "🔍 No specific restaurants found matching your search. Results might be too general or not highly relevant.", None

        response_parts.append(f"🔍 **Relevant Restaurant Safety Records:**\n")

        # Sort restaurants for display (e.g., by name)
        sorted_restaurants_for_display = sorted(restaurants_found_for_text.values(), key=lambda x: x['name'])

        for data in sorted_restaurants_for_display:
            response_parts.append(f"--- **{data['name']}** ({data['borough']}, Cuisine: {data['cuisine']}) ---")
            response_parts.append(f"• **Overall Record:** Recent Grade: `{data['recent_grade']}` | Avg Score: `{data['avg_score']:.1f}` | Total Inspections/Violations in records: `{data['total_violations']}`")

            if data['critical_violations'] > 0:
                response_parts.append(f"• **Critical Violations:** **`{data['critical_violations']}` critical violations found** in records.")
                critical_violations_list = [v for v in data['violations_detail'] if v['critical'] == 'Critical']
                if critical_violations_list:
                    response_parts.append(f"  **Recent Critical Details (Top {min(3, len(critical_violations_list))}):**")
                    for i, violation in enumerate(critical_violations_list[:3]):
                        response_parts.append(f"  - `{violation['date']}`: {violation['description'][:100]}... (Score: {violation['score']})")
                
            elif data['total_violations'] > 0:
                response_parts.append(f"• **No critical violations** found in records.")
                if data['violations_detail']:
                     response_parts.append(f"  **Recent Inspection Details (Top {min(3, len(data['violations_detail']))}):**")
                     for i, violation in enumerate(data['violations_detail'][:3]):
                        response_parts.append(f"  - `{violation['date']}`: {violation['description'][:100]}... (Critical: {violation['critical']}, Score: {violation['score']})")
            else:
                response_parts.append(f"• **No violations found** in records for this restaurant.")
            
            response_parts.append("\n") # Add a blank line for separation

        # --- Create the map visualization ---
        map_object = None
        if map_data:
            map_df = pd.DataFrame(map_data)
            # Ensure latitude and longitude are numeric and drop NaNs
            map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
            map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')
            map_df.dropna(subset=['latitude', 'longitude'], inplace=True) # Drop rows with missing coords

            if not map_df.empty:
                # Use Plotly Express scatter_mapbox for interactive map
                map_object = px.scatter_mapbox(
                    map_df,
                    lat="latitude",
                    lon="longitude",
                    hover_name="name",
                    hover_data={
                        "grade": True,
                        "score": True,
                        "critical_violations": True,
                        "cuisine": True,
                        "borough": True,
                        "latitude": False, # Don't show lat/lon in hover
                        "longitude": False
                    },
                    color="grade", # Color points by grade
                    color_discrete_map={ # Define specific colors for grades
                        "A": "green", "B": "orange", "C": "red",
                        "Not Graded": "gray", "P": "purple", "Z": "blue", # P: Pending, Z: Grade Pending
                        "N": "lightgray" # N: Not Yet Graded
                    },
                    zoom=9, # Adjust initial zoom level
                    height=500,
                    title="Restaurant Locations"
                )
                map_object.update_layout(mapbox_style="open-street-map") # Use OpenStreetMap tiles
                map_object.update_layout(margin={"r":0,"t":40,"l":0,"b":0}) # Adjust margins

        return "\n".join(response_parts), map_object


# =============================================================================
# STREAMLIT APP LAYOUT AND LOGIC
# =============================================================================

# --- Configuration for Data Source ---
# Ensure 'NY_food_safety.csv' is in the SAME DIRECTORY as your app.py in GitHub.
DATA_SOURCE_PATH = 'NY_food_safety.csv'

# --- Requirements for Streamlit Community Cloud ---
# Create a `requirements.txt` file in your GitHub repo with these lines:
# streamlit>=1.20.0
# pandas
# numpy
# scipy
# faiss-cpu
# sentence-transformers
# plotly
# requests
# matplotlib
# seaborn

# Initialize RAG system and assistant globally (with caching)
@st.cache_resource
def get_rag_system_and_assistant(data_source_identifier, sample_size=50000):
    rag_system = NYCFoodSafetyRAG()
    try:
        rag_system.load_data(data_source_identifier, sample_size=sample_size)
        rag_system.build_rag_system(max_docs=sample_size)
        assistant = FoodSafetyAssistant(rag_system)
        return rag_system, assistant
    except Exception as e:
        st.error(f"Failed to initialize the Food Safety Assistant. Please check the data source and try again: {e}")
        st.exception(e)
        return None, None

def main_streamlit_app():
    # Removed 'icon' parameter as it caused a TypeError in some Streamlit versions
    st.set_page_config(page_title="NYC Food Safety Assistant", layout="wide")

    st.title("🍔 NYC Food Safety Assistant 🗽")
    st.markdown("Ask me anything about restaurant safety violations in NYC!")

    # Initialize RAG system and assistant using the local file path
    rag_system, assistant = get_rag_system_and_assistant(DATA_SOURCE_PATH)

    if rag_system is None or assistant is None:
        st.warning("Application could not be initialized. Please ensure 'NY_food_safety.csv' is in the same directory as 'app.py' in your GitHub repository.")
        st.stop() # Stop the app if initialization failed

    # Display sample restaurants and queries in sidebar
    st.sidebar.header("Sample Restaurants & Queries")
    if 'DBA' in rag_system.df.columns and not rag_system.df['DBA'].empty:
        sample_restaurants = rag_system.df['DBA'].drop_duplicates().head(5).tolist()
        st.sidebar.write("Try asking about:")
        for i, restaurant in enumerate(sample_restaurants, 1):
            st.sidebar.markdown(f"- `{restaurant}`")
    else:
        st.sidebar.warning("DBA column not found or is empty in data for sample restaurants.")

    st.sidebar.write("\n**Example Queries:**")
    st.sidebar.markdown("- `Does MCDONALD'S have any critical violations?`")
    st.sidebar.markdown("- `Show me violations for PIZZA HUT`")
    st.sidebar.markdown("- `Analyze critical violations`")
    st.sidebar.markdown("- `Recommend safe restaurants`")
    st.sidebar.markdown("- `What restaurants should I avoid?`")
    st.sidebar.markdown("- `Show me restaurants in MANHATTAN`")
    st.sidebar.markdown("- `Show me a chart of critical violations`")
    st.sidebar.markdown("- `Graph critical rate by borough`")
    st.sidebar.markdown("- `Show grade distribution`")


    # User input
    user_query = st.text_input("Your Question:", "Does MCDONALD'S have any critical violations?", help="Type your question here and press Enter or click 'Ask Assistant'")

    # Initialize session state for last_query if it doesn't exist
    if 'last_query' not in st.session_state:
        st.session_state['last_query'] = ""

    # Check if button is clicked OR if user pressed enter (and query changed)
    if st.button("Ask Assistant") or (user_query and user_query != st.session_state['last_query']):
        st.session_state['last_query'] = user_query # Update last_query

        if not user_query:
            st.warning("Please enter a question.")
            return

        with st.spinner("Processing your request..."):
            response_text, plot_object = assistant.ask(user_query) # Get both text and plot

            st.markdown(response_text) # Display text response

            if plot_object is not None:
                st.plotly_chart(plot_object, use_container_width=True) # Display Plotly chart

    # Optional: Display a small part of the raw data (for debugging/demonstration)
    with st.expander("📊 Peek at Raw Data"):
        if rag_system and rag_system.df is not None:
            st.dataframe(rag_system.df.head())
        else:
            st.info("Data not loaded yet or an error occurred during loading.")


if __name__ == "__main__":
    main_streamlit_app()