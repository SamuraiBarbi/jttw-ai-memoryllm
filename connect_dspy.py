import dspy

def configure_dspy():
    """Configure DSPy to use Ollama local server"""
    try:
        # Set up Ollama local connection
        lm = dspy.OllamaLocal(
            model='llama3.2:3b',
            base_url='http://localhost:11434'
        )
        
        # Configure DSPy with the Ollama language model
        dspy.configure(lm=lm)
        
        print("Successfully configured DSPy with Ollama server")
        print(f"Model: llama3.2:3b")
        print(f"Endpoint: http://localhost:11434")
        return lm
        
    except Exception as e:
        print(f"Error configuring DSPy: {str(e)}")
        return None

def test_llm_response(lm):
    """Test the LLM connection by asking a question"""
    try:
        print("\nTesting LLM connection...")
        question = "Why is the sky blue?"
        print(f"Question: {question}")
        
        response = lm(question)
        print(f"Response: {response}")
        
        return True
    except Exception as e:
        print(f"Error testing LLM: {str(e)}")
        return False

if __name__ == "__main__":
    lm = configure_dspy()
    if lm:
        test_llm_response(lm)
