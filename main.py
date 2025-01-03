from models import Prompt
from mipro import MIPROOptimizer
from copro import COPROOptimizer
from hybrid import HybridOptimizer

def main():
    # Initial prompt
    initial_prompt = Prompt(
        instruction="Explain why the sky appears blue",
        context="The explanation should cover both the scientific principles and be easy to understand",
        examples=[
            "Sunlight contains all colors of the rainbow, but blue light is scattered more by air molecules",
            "This scattering effect, called Rayleigh scattering, makes the sky look blue to our eyes"
        ],
        constraints=[
            "Use simple, non-technical language",
            "Include everyday examples or analogies",
            "Explain the role of sunlight and atmosphere"
        ]
    )
    
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
