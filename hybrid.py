"""
Hybrid Optimizer combining MIPRO and COPRO strategies

This module implements a hybrid optimization strategy that leverages both MIPRO and COPRO optimizers sequentially.
"""

from optimization import PromptOptimizer
from mipro import MIPROOptimizer
from copro import COPROOptimizer
from models import Prompt


class HybridOptimizer(PromptOptimizer):
    """
    Hybrid optimization strategy using both MIPRO and COPRO.
    """

    def hybrid_optimize(self, prompt: Prompt, num_iterations: int = 5) -> Prompt:
        """
        Optimize prompt using both MIPRO and COPRO strategies sequentially.
        
        Args:
            prompt: Initial prompt to optimize
            num_iterations: Number of optimization iterations for MIPRO
            
        Returns:
            Final optimized prompt after both MIPRO and COPRO optimizations
        """
        # First, optimize using MIPRO
        print("\nStarting MIPRO Optimization...")
        mipro_optimizer = MIPROOptimizer()
        optimized_prompt_mipro = mipro_optimizer.optimize(prompt, num_iterations)
        
        # Then, optimize the result using COPRO
        print("\nStarting COPRO Optimization...")
        copro_optimizer = COPROOptimizer()
        optimized_prompt_copro = copro_optimizer.optimize(optimized_prompt_mipro)
        
        return optimized_prompt_copro
