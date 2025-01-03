from typing import List
from pydantic import BaseModel
import dspy

class GenerateContext(dspy.Signature):
    """Signature for generating context from instruction."""
    instruction = dspy.InputField()
    context = dspy.OutputField(desc="Generated context that provides background and guidance")

class GenerateExamples(dspy.Signature):
    """Signature for generating examples from instruction."""
    instruction = dspy.InputField()
    examples = dspy.OutputField(desc="List of 2-3 relevant examples", prefix="Here are some examples:\n")

class GenerateConstraints(dspy.Signature):
    """Signature for generating constraints from instruction."""
    instruction = dspy.InputField()
    constraints = dspy.OutputField(desc="List of 3-4 meaningful constraints", prefix="Here are the constraints:\n")

class ContextGenerator(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, instruction):
        prompt = f"""Given the instruction '{instruction}', generate appropriate context that:
        1. Provides necessary background information
        2. Sets the right scope and depth
        3. Guides the response format and style"""
        
        response = self.lm(prompt)
        return GenerateContext(instruction=instruction, context=response)

class ExamplesGenerator(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, instruction):
        prompt = f"""Given the instruction '{instruction}', generate 2-3 clear examples that:
        1. Illustrate key concepts
        2. Show expected response patterns
        3. Cover different aspects of the topic
        
        Format each example on a new line starting with a hyphen (-)."""
        
        response = self.lm(prompt)
        # Clean up the response and ensure proper formatting
        examples = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith('-'):
                line = f"- {line}"
            if line:
                examples.append(line)
        examples = examples[:3]  # Limit to 3 examples
        return GenerateExamples(instruction=instruction, examples='\n'.join(examples))

class ConstraintsGenerator(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, instruction):
        prompt = f"""Given the instruction '{instruction}', generate 3-4 constraints that:
        1. Ensure response quality
        2. Guide the tone and complexity
        3. Set boundaries for the scope
        4. Specify any requirements or limitations
        
        Format each constraint on a new line starting with a hyphen (-)."""
        
        response = self.lm(prompt)
        # Clean up the response and ensure proper formatting
        constraints = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith('-'):
                line = f"- {line}"
            if line:
                constraints.append(line)
        constraints = constraints[:4]  # Limit to 4 constraints
        return GenerateConstraints(instruction=instruction, constraints='\n'.join(constraints))

class PromptGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        # Get the configured LM from DSPy settings
        lm = dspy.settings.lm
        self.context_generator = ContextGenerator(lm)
        self.examples_generator = ExamplesGenerator(lm)
        self.constraints_generator = ConstraintsGenerator(lm)

    def forward(self, instruction: str) -> dict:
        """Generate context, examples, and constraints based on the instruction."""
        
        # Generate appropriate context
        context_pred = self.context_generator(instruction)
        context = context_pred.context
        
        # Generate relevant examples
        examples_pred = self.examples_generator(instruction)
        examples = examples_pred.examples.split('\n')
        examples = [ex.strip('- ').strip() for ex in examples if ex.strip()]
        
        # Generate meaningful constraints
        constraints_pred = self.constraints_generator(instruction)
        constraints = constraints_pred.constraints.split('\n')
        constraints = [c.strip('- ').strip() for c in constraints if c.strip()]
        
        return {
            "instruction": instruction,
            "context": context,
            "examples": examples,
            "constraints": constraints
        }

class EnhancedPrompt(BaseModel):
    instruction: str
    context: str
    examples: List[str]
    constraints: List[str]
    
    def format(self) -> str:
        """Format the prompt components into a structured string."""
        return f"""Instruction: {self.instruction}

Context: {self.context}

Examples:
{chr(10).join(f'- {example}' for example in self.examples)}

Constraints:
{chr(10).join(f'- {constraint}' for constraint in self.constraints)}"""
