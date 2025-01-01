import dspy
from typing import Optional, Union
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_configuration(model: str, base_url: str) -> bool:
    """Validate the Ollama configuration parameters"""
    if not model or not isinstance(model, str):
        logger.error("Invalid model name")
        return False
    if not base_url or not base_url.startswith(('http://', 'https://')):
        logger.error("Invalid base URL")
        return False
    return True

def configure_dspy(
    model: str = 'llama3.2:3b',
    base_url: str = 'http://localhost:11434',
    max_retries: int = 3,
    retry_delay: int = 2
) -> Optional[dspy.OllamaLocal]:
    """Configure DSPy to use Ollama local server
    
    Args:
        model: The model to use
        base_url: The base URL of the Ollama server
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Configured OllamaLocal instance or None if failed
    """
    if not validate_configuration(model, base_url):
        return None
        
    for attempt in range(max_retries):
        try:
            # Set up Ollama local connection
            lm = dspy.OllamaLocal(
                model=model,
                base_url=base_url
            )
        
            # Configure DSPy with the Ollama language model
            dspy.configure(lm=lm)
            
            logger.info("Successfully configured DSPy with Ollama server")
            logger.info(f"Model: {model}")
            logger.info(f"Endpoint: {base_url}")
            return lm
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            logger.error(f"Failed to configure DSPy after {max_retries} attempts")
            return None

def test_llm_response(lm: dspy.OllamaLocal, timeout: int = 30) -> bool:
    """Test the LLM connection by asking a question"""
    try:
        logger.info("Testing LLM connection...")
        question = "Why is the sky blue?"
        logger.info(f"Question: {question}")
        
        try:
            start_time = time.time()
            response = lm(question)
            elapsed_time = time.time() - start_time
            logger.info(f"Response received in {elapsed_time:.2f}s: {response}")
        except Exception as e:
            logger.error(f"Request timed out after {timeout}s: {str(e)}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error testing LLM: {str(e)}")
        return False

def test_prompts(lm: dspy.OllamaLocal) -> None:
    """Test prompt improvement with multiple predefined prompts
    
    Args:
        lm: Configured language model
    """
    test_prompts = [
        "Why is the sky blue?",
        "Can you tell me about Zeus?",
        "I need help planning a trip to Hawaii",
        "I don't know how to bake a cake",
        "Write a blog post about AI"
    ]
    
    for prompt in test_prompts:
        try:
            logger.info(f"\nTesting prompt improvement for: {prompt}")
            improved_prompt = improve_prompt_iteratively(lm, prompt)
            logger.info(f"Final improved prompt: {improved_prompt}")
        except Exception as e:
            logger.error(f"Error improving prompt '{prompt}': {str(e)}")

def improve_prompt_iteratively(
    lm: dspy.OllamaLocal,
    initial_prompt: str,
    iterations: int = 3,
    timeout: int = 30
) -> str:
    """Iteratively improve a prompt using DSPy's optimization capabilities.
    
    Args:
        lm: Configured language model
        initial_prompt: The starting prompt to improve
        iterations: Number of improvement iterations
        
    Returns:
        The best improved version of the prompt
    """
    try:
        logger.info(f"Improving prompt: {initial_prompt}")
        current_prompt = initial_prompt
        
        for i in range(iterations):
            logger.info(f"Iteration {i+1}/{iterations}")
            
            # Generate improved version using DSPy with timeout
            try:
                start_time = time.time()
                improved_prompt = lm(f"Improve this prompt: {current_prompt}")
                elapsed_time = time.time() - start_time
                logger.info(f"Improved version received in {elapsed_time:.2f}s: {improved_prompt}")
            except Exception as e:
                logger.error(f"Prompt improvement failed: {str(e)}")
                continue
            
            # Evaluate and keep the better version
            current_score = lm(f"Rate this prompt (1-10) with just a number: {current_prompt}")
            improved_score = lm(f"Rate this prompt (1-10) with just a number: {improved_prompt}")
            
            # Handle list responses and extract numeric score
            def extract_score(response):
                if isinstance(response, list):
                    response = response[0]
                # Extract first number from response
                import re
                match = re.search(r'\d+', response)
                return float(match.group()) if match else 0
            
            current_score = extract_score(current_score)
            improved_score = extract_score(improved_score)
            
            if improved_score > current_score:
                logger.info(f"New version is better (score: {improved_score} > {current_score})")
                current_prompt = improved_prompt
            else:
                logger.info(f"Keeping previous version (score: {current_score} >= {improved_score})")
        
        logger.info(f"Final improved prompt: {current_prompt}")
        return current_prompt
        
    except Exception as e:
        logger.error(f"Error improving prompt: {str(e)}")
        return initial_prompt

if __name__ == "__main__":
    lm = configure_dspy()
    if lm:
        test_llm_response(lm)
        test_prompts(lm)
