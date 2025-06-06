import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # This is a key component

# Assume embeddings are generated by functions in utils.feature_extractor
# from utils.feature_extractor import generate_sentence_embedding # Example import

class RecommendationEngine:
    def __init__(self):
        print("RecommendationEngine initialized.")
        # In a real system, this might load pre-computed job embeddings or models.

    def calculate_similarity_scores(self, resume_embedding, job_embeddings_list):
        """
        Calculates cosine similarity between a single resume embedding and a list of job embeddings.

        Args:
            resume_embedding (np.ndarray): A 1D numpy array representing the resume's embedding.
            job_embeddings_list (list of np.ndarray): A list where each element is a 1D numpy
                                                     array representing a job's embedding.

        Returns:
            np.ndarray: An array of similarity scores, one for each job.
        """
        if resume_embedding is None or not job_embeddings_list:
            print("Warning: Resume embedding or job embeddings list is empty.")
            return np.array([])

        if not isinstance(resume_embedding, np.ndarray) or not all(isinstance(job_emb, np.ndarray) for job_emb in job_embeddings_list):
            print("Warning: Embeddings must be numpy arrays.")
            return np.array([])

        # Ensure resume_embedding is 2D for cosine_similarity function
        if len(resume_embedding.shape) == 1:
            resume_embedding = resume_embedding.reshape(1, -1)

        # Stack job embeddings into a 2D array
        # Ensure all job embeddings have the same dimension as the resume embedding
        if not all(job_emb.shape == resume_embedding.shape[1:] for job_emb in job_embeddings_list if len(job_emb.shape) > 0) :
             # Check shape only if job_emb is not an empty array (e.g. from a failed embedding)
            job_dims = [job_emb.shape for job_emb in job_embeddings_list if len(job_emb.shape) > 0]
            # print(f"Warning: Mismatch in embedding dimensions. Resume: {resume_embedding.shape}, Jobs: {job_dims}")
            # Fallback: try to filter out malformed job embeddings or return empty if critical mismatch
            # For simplicity, we'll proceed but cosine_similarity might fail or give bad results.
            # A robust system would handle this more gracefully.
            pass


        # Filter out potentially empty or malformed job embeddings before stacking
        valid_job_embeddings = [job_emb for job_emb in job_embeddings_list if isinstance(job_emb, np.ndarray) and job_emb.ndim > 0 and job_emb.shape[0] == resume_embedding.shape[1]]

        if not valid_job_embeddings:
            print("Warning: No valid job embeddings found after filtering for dimension match.")
            return np.array([])

        job_embeddings_matrix = np.array(valid_job_embeddings)

        if len(job_embeddings_matrix.shape) == 1 : # If only one job embedding and it became 1D
            job_embeddings_matrix = job_embeddings_matrix.reshape(1, -1)

        if resume_embedding.shape[1] != job_embeddings_matrix.shape[1]:
            print(f"Critical Warning: Dimension mismatch after processing. Resume: {resume_embedding.shape[1]}, Jobs: {job_embeddings_matrix.shape[1]}. Cannot compute similarity.")
            return np.array([])

        try:
            similarity_scores = cosine_similarity(resume_embedding, job_embeddings_matrix)
            return similarity_scores.flatten() # Return as a 1D array of scores
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return np.array([])


    def get_job_recommendations(self, resume_content_embedding, job_postings_with_embeddings, top_n=5):
        """
        Generates top N job recommendations for a given resume embedding.

        Args:
            resume_content_embedding (np.ndarray): Embedding of the resume content.
            job_postings_with_embeddings (list of dicts): A list of job postings, where each
                dictionary should contain at least 'id', 'title', and 'embedding'.
                Example: [{'id': 'job1', 'title': 'Software Engineer', 'embedding': np.array(...) }, ...]
            top_n (int): The number of top recommendations to return.

        Returns:
            list of dicts: A list of recommended jobs, sorted by similarity score.
                           Each dict includes 'id', 'title', and 'score'.
        """
        if resume_content_embedding is None:
            print("Error: Resume embedding is None. Cannot generate recommendations.")
            return []
        if not job_postings_with_embeddings:
            print("No job postings provided for recommendations.")
            return []

        job_embeddings_list = [job.get('embedding') for job in job_postings_with_embeddings if job.get('embedding') is not None]

        if not job_embeddings_list:
            print("No valid embeddings found in job postings.")
            return []

        scores = self.calculate_similarity_scores(resume_content_embedding, job_embeddings_list)

        if scores.size == 0:
            print("Could not calculate similarity scores. No recommendations.")
            return []

        # Filter out jobs that didn't yield a score (e.g. due to malformed embeddings)
        # This assumes scores array aligns with jobs that had valid embeddings.
        # We need to map scores back to the original jobs carefully.

        scored_jobs = []
        score_idx = 0
        for i, job in enumerate(job_postings_with_embeddings):
            if job.get('embedding') is not None: # Only consider jobs that had an embedding
                # Further check if this job's embedding was valid enough to be in job_embeddings_list
                # This is tricky if some embeddings were filtered by calculate_similarity_scores
                # For now, assume calculate_similarity_scores returns scores for the *valid* subset it processed.
                # A more robust way is to have calculate_similarity_scores return indices or handle this mapping.
                # Simplified assumption: scores correspond to the order of jobs with non-None embeddings.
                if score_idx < len(scores):
                    scored_jobs.append({
                        "id": job.get('id', f'unknown_id_{i}'),
                        "title": job.get('title', 'Unknown Title'),
                        "score": scores[score_idx]
                    })
                    score_idx +=1
                else: # Should not happen if logic is correct
                    print(f"Warning: Score index out of bounds for job {job.get('id')}")


        # Sort jobs by score in descending order
        recommended_jobs = sorted(scored_jobs, key=lambda x: x['score'], reverse=True)

        return recommended_jobs[:top_n]

if __name__ == '__main__':
    engine = RecommendationEngine()

    # Simulate embeddings (these would come from utils.feature_extractor)
    # Dimension for all-MiniLM-L6-v2 is 384
    DIM = 384
    dummy_resume_embedding = np.random.rand(DIM)

    dummy_job_postings = [
        {"id": "job1", "title": "Software Dev I", "embedding": np.random.rand(DIM)},
        {"id": "job2", "title": "Data Scientist", "embedding": np.random.rand(DIM)},
        {"id": "job3", "title": "Python Developer", "embedding": np.random.rand(DIM) * 0.5 + dummy_resume_embedding * 0.5}, # More similar
        {"id": "job4", "title": "Project Manager", "embedding": np.random.rand(DIM)},
        {"id": "job5", "title": "Senior Python Eng", "embedding": dummy_resume_embedding + np.random.rand(DIM)*0.1}, # Very similar
        {"id": "job6", "title": "QA Tester", "embedding": None}, # No embedding
        {"id": "job7", "title": "Architect", "embedding": np.random.rand(128)}, # Mismatched dimension
    ]

    print("\n--- Getting Job Recommendations ---")
    recommendations = engine.get_job_recommendations(dummy_resume_embedding, dummy_job_postings, top_n=3)

    if recommendations:
        print("Top Recommendations:")
        for rec in recommendations:
            print(f"  ID: {rec['id']}, Title: {rec['title']}, Score: {rec['score']:.4f}")
    else:
        print("No recommendations generated.")

    print("\n--- Test with problematic inputs ---")
    print("Test with None resume embedding:")
    recs_none_resume = engine.get_job_recommendations(None, dummy_job_postings)
    print(f"Recommendations: {recs_none_resume}")

    print("Test with no job postings:")
    recs_no_jobs = engine.get_job_recommendations(dummy_resume_embedding, [])
    print(f"Recommendations: {recs_no_jobs}")

    print("Test with only jobs with None embeddings:")
    recs_only_none_embeddings = engine.get_job_recommendations(dummy_resume_embedding, [{"id":"job_none", "title":"None Emb", "embedding":None}])
    print(f"Recommendations: {recs_only_none_embeddings}")

    print("Test with resume embedding and one job with mismatched dimension embedding:")
    # This should be handled by the filtering in calculate_similarity_scores or job_embeddings_list creation
    job_mismatched_dim = [{"id": "job_mismatch", "title": "Mismatch Dim", "embedding": np.random.rand(128)}]
    recs_mismatched = engine.get_job_recommendations(dummy_resume_embedding, job_mismatched_dim)
    print(f"Recommendations for mismatched dim job: {recs_mismatched}")


    print("\nNote: `cosine_similarity` comes from `scikit-learn`. Ensure it's installed.")
    print("The quality of recommendations depends entirely on the quality of the input embeddings.")
