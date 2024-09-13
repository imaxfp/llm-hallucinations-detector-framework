from sentence_transformers import SentenceTransformer, util

@DeprecationWarning
class SentenceEmbeddingSimilarity:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load a pre-trained model when the class is instantiated
        self.model = SentenceTransformer(model_name)

    def embedding_similarity(self, sentence1, sentence2):
        # Generate embeddings for both sentences
        embedding1 = self.model.encode(sentence1)
        embedding2 = self.model.encode(sentence2)
        
        # Compute cosine similarity
        cosine_sim = util.cos_sim(embedding1, embedding2)
        
        return cosine_sim.item()

if __name__ == "__main__":
    # Create an instance of the class
    similarity_calculator = SentenceEmbeddingSimilarity()
    
    # Call the embedding_similarity method
    result = similarity_calculator.embedding_similarity("test 1", "test 2")
    print(f"Embedding Cosine Similarity: {result}")
