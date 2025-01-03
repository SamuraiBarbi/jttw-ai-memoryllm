import dspy
from typing import List, Optional, Dict, Any
from models import Prompt
import numpy as np
from dataclasses import dataclass
import litellm


@dataclass
class OptimizationMetrics:
    """Stores optimization metrics"""
    relevance_score: float
    clarity_score: float
    constraint_score: float
    reasoning: str  # Added to store chain of thought reasoning
    
    def total_score(self) -> float:
        return (self.relevance_score + self.clarity_score + self.constraint_score) / 3


class PromptOptimizer:
    """Handles prompt optimization using DSPy's capabilities"""
    
    def __init__(self, model: str = "ollama/llama3.2:3b"):
        self.model = model
        self.metric_history = []
        
        # Configure LiteLLM for Ollama
        litellm.set_verbose = False
    
    def optimize(self, prompt: Prompt, metric_fn=None, num_iterations: int = 5) -> Prompt:
        """
        Optimize a prompt using iterative refinement with chain of thought reasoning
        
        Args:
            prompt: The initial prompt to optimize
            metric_fn: Optional function to evaluate prompt quality
            num_iterations: Number of optimization iterations
            
        Returns:
            Optimized prompt
        """
        best_prompt = prompt
        best_metrics = self.evaluate_prompt(prompt)
        print("\nOptimization Process:")
        print("====================")
        
        for i in range(num_iterations):
            print(f"\nIteration {i+1}:")
            # Generate variations using different strategies with reasoning
            variations_with_reasoning = [
                self._add_context(prompt),
                self._refine_instruction(prompt),
                self._generate_new_examples(prompt)
            ]
            
            # Evaluate each variation
            for variation, reasoning in variations_with_reasoning:
                print(f"\nTrying variation with reasoning: {reasoning}")
                metrics = self.evaluate_prompt(variation)
                if metrics.total_score() > best_metrics.total_score():
                    best_prompt = variation
                    best_metrics = metrics
                    print(f"Found better variation! New score: {metrics.total_score():.2f}")
                    print(f"Reasoning: {metrics.reasoning}")
            
            # Track performance
            self.track_performance(best_prompt, best_metrics)
        
        return best_prompt
    
    def hybrid_optimize(self, prompt: Prompt, num_iterations: int = 5) -> Prompt:
        """
        Hybrid optimization using both MIPRO and COPRO strategies.
        
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
    
    def _query_llm_with_cot(self, prompt: str, include_reasoning: bool = True) -> tuple[str, str]:
        """Helper function to query the LLM with chain of thought reasoning"""
        cot_prompt = f"""Let's approach this step-by-step:

1) First, let's understand what we need to do:
{prompt}

2) Let's think about this carefully:
- What are the key elements we need to consider?
- What would make the output most effective?
- How can we ensure it meets all requirements?

3) Based on this thinking, here's my response:

Response: [Your detailed response here]

4) Reasoning: [Explain why this response is effective]

Please provide your response following this format, replacing the placeholders."""

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": cot_prompt}],
            api_base="http://localhost:11434"
        )
        content = response.choices[0].message.content
        
        # Extract response and reasoning
        try:
            response_part = content.split("Response:")[1].split("Reasoning:")[0].strip()
            reasoning_part = content.split("Reasoning:")[1].strip() if include_reasoning else ""
            return response_part, reasoning_part
        except:
            return content, "No explicit reasoning provided"
    
    def _add_context(self, prompt: Prompt) -> tuple[Prompt, str]:
        """Add or refine context using LLM with chain of thought"""
        system_prompt = """Given an instruction, generate relevant context that would help in better understanding or executing the task.
        Consider:
        - What background knowledge would be most helpful?
        - What concepts need clarification?
        - What assumptions should be made explicit?"""
        
        new_context, reasoning = self._query_llm_with_cot(
            f"{system_prompt}\n\nInstruction: {prompt.instruction}"
        )
        
        return (
            Prompt(
                instruction=prompt.instruction,
                context=new_context,
                examples=prompt.examples,
                constraints=prompt.constraints
            ),
            reasoning
        )
    
    def _refine_instruction(self, prompt: Prompt) -> tuple[Prompt, str]:
        """Refine the instruction using LLM with chain of thought"""
        system_prompt = """Given an instruction and its context, generate a more precise and clearer version of the instruction.
        Consider:
        - What ambiguities exist in the current instruction?
        - How can we make it more actionable?
        - What specific details would help?"""
        
        refined_instruction, reasoning = self._query_llm_with_cot(
            f"{system_prompt}\n\nOriginal Instruction: {prompt.instruction}\nContext: {prompt.context or ''}"
        )
        
        return (
            Prompt(
                instruction=refined_instruction,
                context=prompt.context,
                examples=prompt.examples,
                constraints=prompt.constraints
            ),
            reasoning
        )
    
    def _generate_new_examples(self, prompt: Prompt) -> tuple[Prompt, str]:
        """Generate new examples using LLM with chain of thought"""
        if len(prompt.examples) >= 5:  # Limit to 5 examples
            return prompt, "Maximum examples reached"
            
        system_prompt = """Given an instruction and existing examples, generate a new relevant example that is different from the existing ones.
        Consider:
        - What aspects aren't covered by existing examples?
        - How can this new example add value?
        - What edge cases should we consider?"""
        
        new_example, reasoning = self._query_llm_with_cot(
            f"{system_prompt}\n\nInstruction: {prompt.instruction}\nExisting Examples:\n" + 
            "\n".join(f"- {ex}" for ex in prompt.examples)
        )
        
        new_examples = list(prompt.examples)
        new_examples.append(new_example)
        
        return (
            Prompt(
                instruction=prompt.instruction,
                context=prompt.context,
                examples=new_examples,
                constraints=prompt.constraints
            ),
            reasoning
        )
    
    def evaluate_prompt(self, prompt: Prompt) -> OptimizationMetrics:
        """Evaluate prompt quality using LLM with chain of thought reasoning"""
        system_prompt = """Evaluate this prompt based on three criteria and provide detailed reasoning:
        1. Relevance (0-1): How well does it address the task?
        2. Clarity (0-1): How clear and unambiguous is it?
        3. Constraint Adherence (0-1): How well does it follow the constraints?
        
        Think through each criterion carefully and explain your reasoning.
        Then output exactly three numbers between 0 and 1, separated by commas, followed by your reasoning.
        
        Example output format: 
        0.8,0.7,0.9
        Reasoning: The prompt is highly relevant (0.8) because..."""
        
        response, reasoning = self._query_llm_with_cot(
            f"{system_prompt}\n\nPrompt to evaluate:\n{prompt.format()}"
        )
        
        try:
            # Extract scores from the first line and reasoning from subsequent lines
            scores_line = response.split('\n')[0].strip()
            relevance, clarity, constraints = map(float, scores_line.split(','))
        except:
            # Fallback values if parsing fails
            relevance, clarity, constraints = 0.5, 0.5, 0.5
            reasoning = "Failed to parse scores"
            
        return OptimizationMetrics(
            relevance_score=relevance,
            clarity_score=clarity,
            constraint_score=constraints,
            reasoning=reasoning
        )
    
    def track_performance(self, prompt: Prompt, metrics: OptimizationMetrics):
        """Track prompt performance over time with reasoning"""
        self.metric_history.append({
            'prompt': prompt.dict(),
            'metrics': {
                'total_score': metrics.total_score(),
                'relevance': metrics.relevance_score,
                'clarity': metrics.clarity_score,
                'constraints': metrics.constraint_score,
                'reasoning': metrics.reasoning
            }
        })
