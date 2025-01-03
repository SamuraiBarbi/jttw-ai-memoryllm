"""
MIPRO (Mixed Initiative PROmpting) Optimizer

This module implements the MIPRO optimization strategy, which combines:
1. Multi-perspective prompt evaluation
2. Initiative-based optimization
3. Dynamic prompt adjustment
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from optimization import PromptOptimizer, OptimizationMetrics
from models import Prompt
import litellm


@dataclass
class PromptAnalysis:
    """Stores LLM's analysis of prompt variations"""
    strengths: List[str]       # Identified strong points
    weaknesses: List[str]      # Areas for improvement
    suggestions: List[str]     # Specific improvement suggestions
    reasoning: str            # Detailed reasoning for the analysis


class MIPROOptimizer(PromptOptimizer):
    """
    MIPRO optimization strategy that uses multi-perspective LLM analysis
    """
    
    def __init__(self, model: str = "ollama/llama3.2:3b"):
        super().__init__(model)
        self.analysis_history: List[PromptAnalysis] = []
        self.optimization_strategies = {
            'semantic': self._semantic_refinement,
            'structural': self._structural_refinement,
            'contextual': self._contextual_refinement,
            'example': self._example_refinement,
            'perspective': self._perspective_refinement,
            'challenge': self._challenge_refinement,
            'analogy': self._analogy_refinement,
            'misconception': self._misconception_refinement
        }
    
    def optimize(self, prompt: Prompt, num_iterations: int = 5) -> Prompt:
        """
        Optimize prompt using MIPRO strategy with multi-perspective LLM analysis
        
        Args:
            prompt: Initial prompt to optimize
            num_iterations: Number of optimization iterations
            
        Returns:
            Optimized prompt
        """
        best_prompt = prompt
        best_metrics = self.evaluate_prompt(prompt)
        
        print("\nMIPRO Optimization Process:")
        print("==========================")
        
        for i in range(num_iterations):
            print(f"\nIteration {i+1}:")
            
            # Analyze current prompt from multiple perspectives
            analysis = self._analyze_prompt(best_prompt)
            self.analysis_history.append(analysis)
            
            # Generate variations using different strategies
            variations = self._generate_variations(best_prompt, analysis)
            
            # Evaluate and select best variation
            for variation, reasoning in variations:
                metrics = self.evaluate_prompt(variation)
                if metrics.total_score() > best_metrics.total_score():
                    best_prompt = variation
                    best_metrics = metrics
                    print(f"Found better variation! New score: {metrics.total_score():.2f}")
                    print(f"Reasoning: {metrics.reasoning}")
            
            # Track performance
            self.track_performance(best_prompt, best_metrics)
        
        return best_prompt
    
    def _analyze_prompt(self, prompt: Prompt) -> PromptAnalysis:
        """Analyze prompt from multiple perspectives using LLM"""
        system_prompt = """Analyze this prompt from multiple perspectives and provide:
        1. Strengths: What aspects are particularly effective?
        2. Weaknesses: What could be improved?
        3. Suggestions: Specific recommendations for improvement
        4. Reasoning: Detailed explanation of your analysis
        
        Format your response as:
        Strengths:
        - [strength 1]
        - [strength 2]
        
        Weaknesses:
        - [weakness 1]
        - [weakness 2]
        
        Suggestions:
        - [suggestion 1]
        - [suggestion 2]
        
        Reasoning:
        [detailed reasoning]"""
        
        response, _ = self._query_llm_with_cot(
            f"{system_prompt}\n\nPrompt to analyze:\n{prompt.format()}"
        )
        
        # Parse the response into PromptAnalysis components
        try:
            sections = response.split('\n\n')
            strengths = [s.strip('- ') for s in sections[0].split('\n')[1:] if s.strip()]
            weaknesses = [w.strip('- ') for w in sections[1].split('\n')[1:] if w.strip()]
            suggestions = [s.strip('- ') for s in sections[2].split('\n')[1:] if s.strip()]
            reasoning = sections[3].replace('Reasoning:\n', '').strip()
            
            return PromptAnalysis(
                strengths=strengths,
                weaknesses=weaknesses,
                suggestions=suggestions,
                reasoning=reasoning
            )
        except Exception as e:
            print(f"Error parsing analysis: {e}")
            return PromptAnalysis([], [], [], "Analysis parsing failed")
    
    def _generate_variations(self, prompt: Prompt, analysis: PromptAnalysis) -> List[Tuple[Prompt, str]]:
        """Generate multiple prompt variations using different strategies"""
        variations = []
        
        for strategy_name, strategy_fn in self.optimization_strategies.items():
            try:
                variation, reasoning = strategy_fn(prompt, analysis)
                variations.append((variation, f"{strategy_name}: {reasoning}"))
            except Exception as e:
                print(f"Error in strategy {strategy_name}: {e}")
        
        return variations
    
    def _semantic_refinement(self, prompt: Prompt, analysis: PromptAnalysis) -> Tuple[Prompt, str]:
        """Refine prompt focusing on semantic clarity and meaning"""
        system_prompt = """Given the analysis, improve the semantic clarity of this prompt.
        Focus on:
        - Clear and precise language
        - Unambiguous instructions
        - Logical flow of ideas
        Consider these aspects from the analysis:
        Strengths: {strengths}
        Weaknesses: {weaknesses}
        Suggestions: {suggestions}"""
        
        refined_prompt, reasoning = self._query_llm_with_cot(
            f"{system_prompt}\n\nPrompt:\n{prompt.format()}"
        )
        
        return self._parse_refinement(prompt, refined_prompt), reasoning
    
    def _structural_refinement(self, prompt: Prompt, analysis: PromptAnalysis) -> Tuple[Prompt, str]:
        """Refine prompt structure and organization"""
        system_prompt = """Given the analysis, improve the structure of this prompt.
        Focus on:
        - Logical organization
        - Clear sections
        - Effective formatting
        Consider these aspects from the analysis:
        Strengths: {strengths}
        Weaknesses: {weaknesses}
        Suggestions: {suggestions}"""
        
        refined_prompt, reasoning = self._query_llm_with_cot(
            f"{system_prompt}\n\nPrompt:\n{prompt.format()}"
        )
        
        return self._parse_refinement(prompt, refined_prompt), reasoning
    
    def _contextual_refinement(self, prompt: Prompt, analysis: PromptAnalysis) -> Tuple[Prompt, str]:
        """Refine prompt context and background information"""
        system_prompt = """Given the analysis, improve the context of this prompt.
        Focus on:
        - Relevant background information
        - Necessary assumptions
        - Scope clarification
        Consider these aspects from the analysis:
        Strengths: {strengths}
        Weaknesses: {weaknesses}
        Suggestions: {suggestions}"""
        
        refined_prompt, reasoning = self._query_llm_with_cot(
            f"{system_prompt}\n\nPrompt:\n{prompt.format()}"
        )
        
        return self._parse_refinement(prompt, refined_prompt), reasoning
    
    def _example_refinement(self, prompt: Prompt, analysis: PromptAnalysis) -> Tuple[Prompt, str]:
        """Refine prompt examples"""
        system_prompt = """Given the analysis, improve the examples in this prompt.
        Focus on:
        - Relevance to the task
        - Clarity of illustration
        - Coverage of edge cases
        Consider these aspects from the analysis:
        Strengths: {strengths}
        Weaknesses: {weaknesses}
        Suggestions: {suggestions}"""
        
        refined_prompt, reasoning = self._query_llm_with_cot(
            f"{system_prompt}\n\nPrompt:\n{prompt.format()}"
        )
        
        return self._parse_refinement(prompt, refined_prompt), reasoning
    
    def _perspective_refinement(self, prompt: Prompt, analysis: PromptAnalysis) -> tuple[Prompt, str]:
        """Add multiple perspectives to enhance understanding"""
        system_prompt = """Given this prompt, generate alternative perspectives that could enhance understanding.
        Consider:
        - Different learning styles (visual, auditory, kinesthetic)
        - Various backgrounds (technical, non-technical)
        - Different age groups
        - Real-world applications
        
        Format your response as a new instruction that incorporates these perspectives."""
        
        refined_prompt, reasoning = self._query_llm_with_cot(
            f"{system_prompt}\n\nOriginal Prompt:\n{prompt.format()}"
        )
        
        return self._parse_refinement(prompt, refined_prompt), reasoning

    def _challenge_refinement(self, prompt: Prompt, analysis: PromptAnalysis) -> tuple[Prompt, str]:
        """Add challenge questions to promote deeper understanding"""
        system_prompt = """Generate thought-provoking questions that challenge common assumptions about this topic.
        Consider:
        - Edge cases
        - Counter-intuitive aspects
        - Common misconceptions
        - Real-world exceptions
        
        Format your response as additional context that addresses these challenges."""
        
        refined_prompt, reasoning = self._query_llm_with_cot(
            f"{system_prompt}\n\nOriginal Prompt:\n{prompt.format()}"
        )
        
        return self._parse_refinement(prompt, refined_prompt), reasoning

    def _analogy_refinement(self, prompt: Prompt, analysis: PromptAnalysis) -> tuple[Prompt, str]:
        """Generate powerful analogies to simplify complex concepts"""
        system_prompt = """Create meaningful analogies that make this concept more accessible.
        Consider:
        - Everyday experiences
        - Common objects
        - Familiar processes
        - Universal experiences
        
        Format your response as new examples using these analogies."""
        
        refined_prompt, reasoning = self._query_llm_with_cot(
            f"{system_prompt}\n\nOriginal Prompt:\n{prompt.format()}"
        )
        
        return self._parse_refinement(prompt, refined_prompt), reasoning

    def _misconception_refinement(self, prompt: Prompt, analysis: PromptAnalysis) -> tuple[Prompt, str]:
        """Address and correct common misconceptions"""
        system_prompt = """Identify and address common misconceptions about this topic.
        Consider:
        - Popular myths
        - Oversimplifications
        - Outdated beliefs
        - Partial understandings
        
        Format your response as additional context that clarifies these misconceptions."""
        
        refined_prompt, reasoning = self._query_llm_with_cot(
            f"{system_prompt}\n\nOriginal Prompt:\n{prompt.format()}"
        )
        
        return self._parse_refinement(prompt, refined_prompt), reasoning
    
    def _parse_refinement(self, original_prompt: Prompt, refinement: str) -> Prompt:
        """Parse refinement suggestion into a new Prompt object"""
        try:
            # Initialize components with original values
            components = {
                'instruction': original_prompt.instruction,
                'context': original_prompt.context,
                'examples': original_prompt.examples.copy(),
                'constraints': original_prompt.constraints.copy() if original_prompt.constraints else []
            }
            
            # Extract new components if present in refinement
            if "Instruction:" in refinement:
                instruction_text = refinement.split("Instruction:")[1]
                instruction_end = (
                    instruction_text.find("Context:") if "Context:" in instruction_text
                    else instruction_text.find("Examples:") if "Examples:" in instruction_text
                    else instruction_text.find("Constraints:") if "Constraints:" in instruction_text
                    else len(instruction_text)
                )
                components['instruction'] = instruction_text[:instruction_end].strip()
            
            if "Context:" in refinement:
                context_text = refinement.split("Context:")[1]
                context_end = (
                    context_text.find("Examples:") if "Examples:" in context_text
                    else context_text.find("Constraints:") if "Constraints:" in context_text
                    else context_text.find("Instruction:") if "Instruction:" in context_text
                    else len(context_text)
                )
                context = context_text[:context_end].strip()
                if context and not context.isspace() and context != "**":
                    components['context'] = context
            
            if "Examples:" in refinement:
                examples_text = refinement.split("Examples:")[1]
                examples_end = (
                    examples_text.find("Constraints:") if "Constraints:" in examples_text
                    else examples_text.find("Instruction:") if "Instruction:" in examples_text
                    else examples_text.find("Context:") if "Context:" in examples_text
                    else len(examples_text)
                )
                examples = examples_text[:examples_end].strip()
                if examples and not examples.isspace() and examples != "**":
                    components['examples'] = [
                        ex.strip("- ").strip()
                        for ex in examples.split("\n")
                        if ex.strip("- ").strip() and ex.strip("- ").strip() != "**"
                    ]
            
            if "Constraints:" in refinement:
                constraints_text = refinement.split("Constraints:")[1]
                constraints_end = (
                    constraints_text.find("Instruction:") if "Instruction:" in constraints_text
                    else constraints_text.find("Context:") if "Context:" in constraints_text
                    else constraints_text.find("Examples:") if "Examples:" in constraints_text
                    else len(constraints_text)
                )
                constraints = constraints_text[:constraints_end].strip()
                if constraints and not constraints.isspace() and constraints != "**":
                    components['constraints'] = [
                        c.strip("- ").strip()
                        for c in constraints.split("\n")
                        if c.strip("- ").strip() and c.strip("- ").strip() != "**"
                    ]
            
            # Create new prompt with updated components
            return Prompt(
                instruction=components['instruction'],
                context=components['context'],
                examples=components['examples'],
                constraints=components['constraints']
            )
            
        except Exception as e:
            print(f"Error parsing refinement: {e}")
            return original_prompt
