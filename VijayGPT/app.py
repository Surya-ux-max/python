import os
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class VijayGPT:
    def __init__(self):
        print("\n" + "="*60)
        print("         Welcome to VijayGPT - Terminal Edition")
        print("="*60)
        print("Loading AI model... Please wait...")
        
        # Initialize the sentence transformer model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[SUCCESS] Model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            print("Please install sentence-transformers: pip install sentence-transformers")
            sys.exit(1)
        
        # Knowledge base about Thalapathy Vijay
        self.knowledge_base = [
            "Thalapathy Vijay is a popular Indian actor known for his versatile roles and charismatic screen presence.",
            "He has a massive fan following and is celebrated for his contributions to Tamil cinema.",
            "He is known for his philanthropic work and has a significant impact on the Tamil film industry.",
            "He is a fabulous dancer who is known for his energetic performances and unique dance style.",
            "He is also known for his singing abilities and has sung several songs in his movies.",
            "He started TVK (Tamilaga Vettri Kazhagam) in 2024 and conducted a meeting in Vikravandi where more than 3 lakh people attended.",
            "Unfortunately, in 2025 due to political tensions, a stampede happened in Karur and 41 people died.",
            "Despite the tragedy, he is still loved by his fans and continues to be a major influence in the industry.",
            "He rose above the tragedy and continues to inspire his fans with his resilience and dedication to his craft.",
            "He campaigned across entire Tamil Nadu in 2026 Assembly elections.",
            "The number of people who attended his meetings was more than 5 lakh.",
            "People approached from every direction to see him and his charisma was at its peak.",
            "In 2026 elections, Thalapathy Vijay's party won 108 seats and he became the Chief Minister of Tamil Nadu.",
            "He is a visionary leader who has brought positive changes to the state.",
            "Vijay has acted in over 60 films and is one of the highest-paid actors in South India.",
            "He is known for his action sequences and emotional performances in movies.",
            "His popular movies include Thuppakki, Kaththi, Mersal, Bigil, and Master.",
            "He often plays roles that have social messages and fight against corruption."
        ]
        
        # Pre-compute embeddings for the knowledge base
        print("Processing knowledge base...")
        self.knowledge_embeddings = self.model.encode(self.knowledge_base)
        print("[SUCCESS] Knowledge base ready!")
        print("\nYou can now ask questions about Thalapathy Vijay!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")
    
    def find_best_answer(self, question):
        """Find the most relevant answer from the knowledge base"""
        try:
            # Encode the user question
            question_embedding = self.model.encode([question])
            
            # Calculate similarities
            similarities = cosine_similarity(question_embedding, self.knowledge_embeddings)
            
            # Find the best match
            best_match_index = similarities.argmax()
            confidence = similarities[0][best_match_index]
            
            return self.knowledge_base[best_match_index], confidence
        
        except Exception as e:
            return f"Error processing question: {e}", 0.0
    
    def generate_response(self, question, context, confidence):
        """Generate a response based on the context and confidence"""
        if confidence < 0.3:
            return ("I'm not sure about that. Could you ask something more specific about "
                   "Thalapathy Vijay's career, movies, politics, or personal life?")
        
        # Create a more natural response
        response = f"Based on what I know: {context}"
        
        if confidence > 0.7:
            response += "\n\nThis information seems highly relevant to your question."
        elif confidence > 0.5:
            response += "\n\nThis might be related to what you're asking about."
        
        return response
    
    def run(self):
        """Main conversation loop"""
        while True:
            try:
                # Get user input
                question = input("\nAsk me about Thalapathy Vijay: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\nThank you for using VijayGPT! Goodbye!")
                    break
                
                # Handle empty input
                if not question:
                    print("Please ask a question!")
                    continue
                
                # Show processing indicator
                print("Searching for answer...")
                
                # Find the best answer
                context, confidence = self.find_best_answer(question)
                
                # Generate and display response
                response = self.generate_response(question, context, confidence)
                print(f"\nVijayGPT: {response}")
                
                # Show confidence level for debugging
                print(f"\nConfidence: {confidence:.2f}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for using VijayGPT!")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                print("Please try asking your question again.")

def main():
    """Main function to run VijayGPT"""
    try:
        # Create and run VijayGPT instance
        vijay_gpt = VijayGPT()
        vijay_gpt.run()
    except Exception as e:
        print(f"Failed to start VijayGPT: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()