from prompt_generator import PromptGenerator, EnhancedPrompt
from mipro import MIPROOptimizer
from copro import COPROOptimizer
from hybrid import HybridOptimizer
import dspy
from dspy.teleprompt import BootstrapFewShot
import requests

class OllamaLLM(dspy.LM):
    def __init__(self, model="llama3.2:3b", url="http://localhost:11434"):
        super().__init__(model=model)
        self.url = url
        self.endpoint = f"{url}/api/generate"

    def basic_request(self, prompt, **kwargs):
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error calling Ollama: {str(e)}")
            return ""

    def __call__(self, prompt, **kwargs):
        response = self.basic_request(prompt, **kwargs)
        return response

def main():
    # Configure DSPy with local Ollama LLM
    ollama = OllamaLLM(model="llama3.2:3b")
    dspy.settings.configure(lm=ollama)
    
    # Initialize prompt generator
    prompt_generator = PromptGenerator()
    
    # Generate enhanced prompt components from just the instruction
    instruction = "Explain why the sky appears blue"
    prompt_components = prompt_generator.forward(instruction)
    
    # Create the initial prompt using generated components
    initial_prompt = EnhancedPrompt(**prompt_components)
    
    print("\nInitial Prompt:")
    print("-" * 50)
    print(initial_prompt.format())
    print("-" * 50)
    
    # Initialize MIPRO optimizer
    mipro_optimizer = MIPROOptimizer()
    
    # Optimize prompt using MIPRO
    optimized_prompt_mipro = mipro_optimizer.optimize(initial_prompt)
    
    print("\nFinal Optimization Results (MIPRO):")
    print("=" * 50)
    print("\nOptimized Prompt:")
    print("-" * 50)
    print(optimized_prompt_mipro.format())
    print("-" * 50)
    
    # Initialize COPRO optimizer
    copro_optimizer = COPROOptimizer()
    
    # Optimize prompt using COPRO
    optimized_prompt_copro = copro_optimizer.optimize(initial_prompt)
    
    print("\nFinal Optimization Results (COPRO):")
    print("=" * 50)
    print("\nOptimized Prompt:")
    print("-" * 50)
    print(optimized_prompt_copro.format())
    print("-" * 50)
    
    # Initialize hybrid optimizer
    hybrid_optimizer = HybridOptimizer()
    
    # Optimize prompt using hybrid method
    optimized_prompt_hybrid = hybrid_optimizer.hybrid_optimize(initial_prompt)
    
    print("\nFinal Optimization Results (Hybrid):")
    print("=" * 50)
    print("\nOptimized Prompt:")
    print("-" * 50)
    print(optimized_prompt_hybrid.format())
    print("-" * 50)

if __name__ == "__main__":
    main()
