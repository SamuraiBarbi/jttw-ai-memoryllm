"""
COPRO (COntextual PROmpting) Optimizer

This module implements the COPRO optimization strategy, which focuses on:
1. Context-aware refinement
2. Semantic understanding
3. Adaptive prompt generation based on context
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass
from optimization import PromptOptimizer
from models import Prompt


@dataclass
class ContextAnalysis:
    """Stores detailed context analysis results"""
    key_elements: List[str]
    missing_elements: List[str]
    improvement_areas: List[str]
    semantic_topics: List[str]
    context_score: float
    reasoning: str
    audience_factors: List[str]
    complexity_level: str
    knowledge_prerequisites: List[str]


class COPROOptimizer(PromptOptimizer):
    """
    COPRO optimization strategy that uses context-aware refinement
    """
    
    def __init__(self, model: str = "ollama/llama3.2:3b"):
        super().__init__(model)
        self.variation_strategies = {
            'semantic_enhancement': self._semantic_enhancement_strategy,
            'topic_expansion': self._topic_expansion_strategy,
            'context_refinement': self._context_refinement_strategy,
            'knowledge_integration': self._knowledge_integration_strategy,
            'audience_adaptation': self._audience_adaptation_strategy,
            'complexity_adjustment': self._complexity_adjustment_strategy,
            'prerequisite_integration': self._prerequisite_integration_strategy,
            'interactive_elements': self._interactive_elements_strategy
        }

    def optimize(self, prompt: Prompt, num_iterations: int = 5) -> Prompt:
        """
        Optimize prompt using COPRO strategy with context-aware refinement
        
        Args:
            prompt: Initial prompt to optimize
            num_iterations: Number of optimization iterations
            
        Returns:
            Optimized prompt
        """
        best_prompt = prompt
        best_metrics = self.evaluate_prompt(prompt)
        
        print("\nCOPRO Optimization Process:")
        print("==========================")
        
        for i in range(num_iterations):
            print(f"\nIteration {i+1}:")
            
            # Perform detailed context analysis
            context_analysis = self._analyze_context(best_prompt)
            print(f"Context Score: {context_analysis.context_score:.2f}")
            
            # Generate variations using multiple strategies
            variations = self._generate_contextual_variations(best_prompt, context_analysis)
            
            # Evaluate and select best variation
            for variation, strategy_name in variations:
                metrics = self.evaluate_prompt(variation)
                if metrics.total_score() > best_metrics.total_score():
                    best_prompt = variation
                    best_metrics = metrics
                    print(f"Found better variation using {strategy_name}! New score: {metrics.total_score():.2f}")
                    print(f"Reasoning: {metrics.reasoning}")
            
            # Track performance
            self.track_performance(best_prompt, best_metrics)
        
        return best_prompt
    
    def _analyze_context(self, prompt: Prompt) -> ContextAnalysis:
        """Perform detailed context analysis using semantic understanding"""
        system_prompt = """Perform a comprehensive analysis of this prompt's context. Consider:

1. Key Elements Analysis:
- What are the main concepts present?
- What contextual information is provided?
- What assumptions are made explicit?

2. Missing Elements:
- What important context is missing?
- What assumptions need clarification?
- What background information would help?

3. Improvement Areas:
- Where could the context be more precise?
- What ambiguities exist?
- What could enhance understanding?

4. Semantic Topics:
- What are the main topics covered?
- What related concepts could be relevant?
- What domain knowledge is important?

5. Audience Factors:
- What background knowledge is assumed?
- What learning styles are addressed?
- What engagement levels are targeted?

6. Complexity Assessment:
- What is the current complexity level?
- Is it appropriate for the target audience?
- What adjustments might be needed?

7. Knowledge Prerequisites:
- What foundational concepts are needed?
- What skills should be in place?
- What resources might be helpful?

Format your response exactly as:
Key Elements:
- [element1]
- [element2]

Missing Elements:
- [missing1]
- [missing2]

Improvement Areas:
- [improvement1]
- [improvement2]

Semantic Topics:
- [topic1]
- [topic2]

Audience Factors:
- [factor1]
- [factor2]

Complexity Level: [basic/intermediate/advanced]

Knowledge Prerequisites:
- [prerequisite1]
- [prerequisite2]

Context Score: [0-1]

Reasoning: [detailed explanation]"""
        
        response, _ = self._query_llm_with_cot(
            f"{system_prompt}\n\nPrompt to Analyze:\n{prompt.format()}"
        )
        
        try:
            sections = response.split('\n\n')
            
            # Extract all components
            key_elements = [e.strip('- ') for e in sections[0].split('\n')[1:] if e.strip()]
            missing_elements = [e.strip('- ') for e in sections[1].split('\n')[1:] if e.strip()]
            improvement_areas = [e.strip('- ') for e in sections[2].split('\n')[1:] if e.strip()]
            semantic_topics = [e.strip('- ') for e in sections[3].split('\n')[1:] if e.strip()]
            audience_factors = [e.strip('- ') for e in sections[4].split('\n')[1:] if e.strip()]
            complexity_level = sections[5].split(':')[1].strip()
            knowledge_prerequisites = [e.strip('- ') for e in sections[6].split('\n')[1:] if e.strip()]
            score = float(sections[7].split(':')[1].strip())
            reasoning = sections[8].split(':')[1].strip()
            
            return ContextAnalysis(
                key_elements=key_elements,
                missing_elements=missing_elements,
                improvement_areas=improvement_areas,
                semantic_topics=semantic_topics,
                context_score=score,
                reasoning=reasoning,
                audience_factors=audience_factors,
                complexity_level=complexity_level,
                knowledge_prerequisites=knowledge_prerequisites
            )
        except Exception as e:
            print(f"Error parsing context analysis: {e}")
            return ContextAnalysis([], [], [], [], 0.5, "Analysis parsing failed", [], "intermediate", [])

    def _generate_contextual_variations(self, prompt: Prompt, analysis: ContextAnalysis) -> List[Tuple[Prompt, str]]:
        """Generate variations using multiple sophisticated strategies"""
        variations = []
        
        # Apply each variation strategy
        for strategy_name, strategy_fn in self.variation_strategies.items():
            try:
                variation = strategy_fn(prompt, analysis)
                variations.append((variation, strategy_name))
            except Exception as e:
                print(f"Error in strategy {strategy_name}: {e}")
        
        return variations
    
    def _semantic_enhancement_strategy(self, prompt: Prompt, analysis: ContextAnalysis) -> Prompt:
        """Enhance prompt by incorporating semantic understanding"""
        semantic_context = "\n".join([
            "Semantic Context:",
            "- Key Concepts: " + ", ".join(analysis.key_elements),
            "- Related Topics: " + ", ".join(analysis.semantic_topics),
            prompt.context if prompt.context else ""
        ])
        
        return Prompt(
            instruction=prompt.instruction,
            context=semantic_context,
            examples=prompt.examples,
            constraints=prompt.constraints
        )
    
    def _topic_expansion_strategy(self, prompt: Prompt, analysis: ContextAnalysis) -> Prompt:
        """Expand prompt by incorporating related topics"""
        system_prompt = f"""Given these topics: {', '.join(analysis.semantic_topics)}
        Generate a brief, relevant explanation that connects them to the main instruction: {prompt.instruction}"""
        
        expansion, _ = self._query_llm_with_cot(system_prompt)
        
        expanded_context = f"{prompt.context}\n\nExpanded Context:\n{expansion}" if prompt.context else expansion
        
        return Prompt(
            instruction=prompt.instruction,
            context=expanded_context,
            examples=prompt.examples,
            constraints=prompt.constraints
        )
    
    def _context_refinement_strategy(self, prompt: Prompt, analysis: ContextAnalysis) -> Prompt:
        """Refine context by addressing improvement areas"""
        refinements = []
        
        for area in analysis.improvement_areas:
            system_prompt = f"""Given this improvement area: {area}
            Provide a clear, concise refinement for the context that addresses this specific issue."""
            
            refinement, _ = self._query_llm_with_cot(system_prompt)
            refinements.append(refinement)
        
        refined_context = f"{prompt.context}\n\nRefinements:\n" + "\n".join(f"- {r}" for r in refinements)
        
        return Prompt(
            instruction=prompt.instruction,
            context=refined_context,
            examples=prompt.examples,
            constraints=prompt.constraints
        )
    
    def _knowledge_integration_strategy(self, prompt: Prompt, analysis: ContextAnalysis) -> Prompt:
        """Integrate missing knowledge elements"""
        if not analysis.missing_elements:
            return prompt
            
        system_prompt = f"""Given these missing elements: {', '.join(analysis.missing_elements)}
        Provide concise, relevant information that fills these knowledge gaps while maintaining clarity."""
        
        knowledge, _ = self._query_llm_with_cot(system_prompt)
        
        enhanced_context = f"{prompt.context}\n\nAdditional Context:\n{knowledge}" if prompt.context else knowledge
        
        return Prompt(
            instruction=prompt.instruction,
            context=enhanced_context,
            examples=prompt.examples,
            constraints=prompt.constraints
        )

    def _audience_adaptation_strategy(self, prompt: Prompt, analysis: ContextAnalysis) -> Prompt:
        """Adapt the prompt for different audience types"""
        audience_prompt = f"""Given these audience factors: {', '.join(analysis.audience_factors)}
        Adapt the prompt to better serve different learning styles and backgrounds while maintaining core content."""
        
        adaptation, _ = self._query_llm_with_cot(audience_prompt)
        
        return Prompt(
            instruction=prompt.instruction,
            context=f"{prompt.context}\n\nAudience-Specific Guidance:\n{adaptation}",
            examples=prompt.examples,
            constraints=prompt.constraints
        )

    def _complexity_adjustment_strategy(self, prompt: Prompt, analysis: ContextAnalysis) -> Prompt:
        """Adjust the complexity level of the prompt"""
        adjustment_prompt = f"""Current complexity level: {analysis.complexity_level}
        Provide variations of the key concepts at different complexity levels while maintaining accuracy."""
        
        adjustment, _ = self._query_llm_with_cot(adjustment_prompt)
        
        return Prompt(
            instruction=prompt.instruction,
            context=f"{prompt.context}\n\nMulti-level Explanation:\n{adjustment}",
            examples=prompt.examples,
            constraints=prompt.constraints
        )

    def _prerequisite_integration_strategy(self, prompt: Prompt, analysis: ContextAnalysis) -> Prompt:
        """Integrate prerequisite knowledge seamlessly"""
        if not analysis.knowledge_prerequisites:
            return prompt
            
        prereq_prompt = f"""Given these prerequisites: {', '.join(analysis.knowledge_prerequisites)}
        Create a brief, clear explanation of essential background knowledge needed for full understanding."""
        
        prerequisites, _ = self._query_llm_with_cot(prereq_prompt)
        
        return Prompt(
            instruction=prompt.instruction,
            context=f"Background Knowledge:\n{prerequisites}\n\n{prompt.context}",
            examples=prompt.examples,
            constraints=prompt.constraints
        )

    def _interactive_elements_strategy(self, prompt: Prompt, analysis: ContextAnalysis) -> Prompt:
        """Add interactive elements to enhance engagement"""
        interactive_prompt = """Generate interactive elements that encourage active learning:
        - Thought experiments
        - Prediction exercises
        - Real-world observations
        - Self-check questions"""
        
        elements, _ = self._query_llm_with_cot(interactive_prompt)
        
        return Prompt(
            instruction=prompt.instruction,
            context=f"{prompt.context}\n\nInteractive Elements:\n{elements}",
            examples=prompt.examples,
            constraints=prompt.constraints
        )
