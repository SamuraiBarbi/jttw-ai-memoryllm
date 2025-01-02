import dspy
from typing import Optional, Union, List, Dict, Any
import time
import logging
import numpy as np
from dataclasses import dataclass

@dataclass
class PromptExample:
    """Dataclass for storing prompt optimization examples"""
    input_text: str
    expected_output: str
    metadata: Dict[Any, Any] = None

class PromptSignature(dspy.Signature):
    """Defines the input/output structure for prompt optimization"""
    original_prompt = dspy.InputField(desc="Original prompt to be improved")
    context = dspy.InputField(desc="Additional context or constraints")
    improved_prompt = dspy.OutputField(desc="Optimized version of the prompt")
    reasoning = dspy.OutputField(desc="Explanation of improvements made")

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
) -> Optional[dspy.LM]:
    """Configure DSPy to use the language model
    
    Args:
        model: The model to use
        base_url: The base URL of the Ollama server
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Configured LM instance or None if failed
    """
    if not validate_configuration(model, base_url):
        return None
        
    for attempt in range(max_retries):
        try:
            # Set up language model connection
            lm = dspy.LM(
                model=f'ollama/{model}',
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

def test_llm_response(lm: dspy.LM, timeout: int = 30) -> bool:
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

def test_prompts(lm: dspy.LM) -> None:
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

class PromptOptimizer(dspy.Module):
    def __init__(self, model_name: str = "llama3.2:3b"):
        super().__init__()
        self.improve_prompt = dspy.ChainOfThought(PromptSignature)
        self.model_name = model_name
        self.history = []
        
    def forward(self, prompt: str, context: str = "") -> dict:
        """Optimize a prompt using Chain of Thought reasoning
        
        Args:
            prompt: The prompt to optimize
            context: Additional context or constraints
            
        Returns:
            Dictionary containing improved prompt and reasoning
        """
        try:
            prediction = self.improve_prompt(
                original_prompt=prompt,
                context=context
            )
            result = {
                "improved_prompt": prediction.improved_prompt,
                "reasoning": prediction.reasoning
            }
            self.history.append({
                "original_prompt": prompt,
                "result": result
            })
            return result
        except Exception as e:
            logger.error(f"Prompt optimization failed: {str(e)}")
            return {
                "improved_prompt": prompt,
                "reasoning": "Optimization failed"
            }

def improve_prompt_iteratively(
    lm: dspy.LM,
    initial_prompt: str,
    examples: List[PromptExample] = None,
    max_iterations: int = 5,
    min_score_improvement: float = 0.5,
    timeout: int = 30
) -> str:
    """Iteratively improve a prompt using Bayesian optimization.
    
    Args:
        lm: Configured language model
        initial_prompt: The starting prompt to improve
        examples: List of PromptExample instances for evaluation
        max_iterations: Maximum number of improvement iterations
        min_score_improvement: Minimum score improvement required to continue iterations
        timeout: Timeout for each LLM request
        
    Returns:
        The best improved version of the prompt
    """
    base_optimizer = PromptOptimizer()
    bayesian_optimizer = BayesianPromptOptimizer(base_optimizer)
    
    # Create default examples if none provided
    if examples is None:
        examples = [
            PromptExample(
                input_text=initial_prompt,
                expected_output="",  # Placeholder, should be filled with expected outputs
                metadata={"context": "general"}
            )
        ]
    
    try:
        logger.info(f"Improving prompt: {initial_prompt}")
        best_prompt = bayesian_optimizer.optimize(
            initial_prompt=initial_prompt,
            examples=examples,
            n_iterations=max_iterations
        )
        
        # Evaluate final prompt
        best_score = evaluate_prompt(lm, best_prompt)
        logger.info(f"Final improved prompt (score: {best_score:.2f}): {best_prompt}")
        
        return best_prompt
        
    except Exception as e:
        logger.error(f"Error improving prompt: {str(e)}")
        return initial_prompt

class BayesianPromptOptimizer:
    def __init__(self, base_optimizer: PromptOptimizer):
        self.base_optimizer = base_optimizer
        self.optimization_history = []
        
    def quality_metric(self, prediction: str, gold: str) -> float:
        """Compute quality score between 0 and 1"""
        # Implement your quality metric here
        return np.random.random()  # Placeholder
    
    def generate_candidates(self, prompt: str, n_candidates: int = 5) -> List[str]:
        """Generate candidate prompts using base optimizer"""
        candidates = []
        for _ in range(n_candidates):
            result = self.base_optimizer.forward(prompt)
            candidates.append(result["improved_prompt"])
        return candidates
    
    def optimize(self, 
                initial_prompt: str,
                examples: List[PromptExample],
                n_iterations: int = 10,
                n_candidates: int = 5) -> str:
        current_best = initial_prompt
        best_score = 0
        
        for iteration in range(n_iterations):
            # Generate candidates
            candidates = self.generate_candidates(current_best, n_candidates)
            
            # Evaluate candidates
            scores = []
            for candidate in candidates:
                candidate_scores = []
                for example in examples:
                    try:
                        result = self.base_optimizer.forward(
                            candidate, 
                            context=str(example.metadata) if example.metadata else ""
                        )
                        score = self.quality_metric(
                            result["improved_prompt"],
                            example.expected_output
                        )
                        candidate_scores.append(score)
                    except Exception as e:
                        candidate_scores.append(0)
                        
                avg_score = np.mean(candidate_scores)
                scores.append(avg_score)
                
                if avg_score > best_score:
                    current_best = candidate
                    best_score = avg_score
            
            self.optimization_history.append({
                'iteration': iteration,
                'best_score': best_score,
                'best_prompt': current_best
            })
            
        return current_best

def evaluate_prompt(lm: dspy.LM, prompt: str) -> float:
    """Evaluate a prompt using multiple criteria and return a composite score
    
    Args:
        lm: Configured language model
        prompt: The prompt to evaluate
        
    Returns:
        Composite score between 0 and 10
    """
    try:
        def extract_score(response):
            if isinstance(response, list):
                response = response[0]
            # Extract first number from response
            import re
            match = re.search(r'\d+', str(response))
            return float(match.group()) if match else 0.0
            
        # Get scores for different aspects
        clarity_score = extract_score(lm(f"Rate the clarity of this prompt (1-10): {prompt}"))
        specificity_score = extract_score(lm(f"Rate the specificity of this prompt (1-10): {prompt}"))
        engagement_score = extract_score(lm(f"Rate the engagement of this prompt (1-10): {prompt}"))
        
        # Calculate weighted composite score
        composite_score = (
            clarity_score * 0.4 +
            specificity_score * 0.4 +
            engagement_score * 0.2
        )
        
        logger.debug(f"Prompt evaluation scores - Clarity: {clarity_score:.2f}, Specificity: {specificity_score:.2f}, Engagement: {engagement_score:.2f}, Composite: {composite_score:.2f}")
        return composite_score
        
    except Exception as e:
        logger.error(f"Prompt evaluation failed: {str(e)}")
        return 0.0

if __name__ == "__main__":
    lm = configure_dspy()
    if lm:
        test_llm_response(lm)
        test_prompts(lm)
