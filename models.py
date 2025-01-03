from pydantic import BaseModel, Field
from typing import List, Optional


class Prompt(BaseModel):
    """Base model for prompt structure"""
    instruction: str = Field(..., description="The main instruction or question")
    context: Optional[str] = Field(None, description="Additional context for the prompt")
    examples: List[str] = Field(default_factory=list, description="Few-shot examples")
    constraints: Optional[List[str]] = Field(default_factory=list, description="Any constraints or requirements")
    
    def format(self) -> str:
        """Format the prompt for LLM consumption"""
        parts = []
        
        # Add context if available
        if self.context:
            parts.append(f"Context:\n{self.context}\n")
        
        # Add examples if available
        if self.examples:
            parts.append("Examples:")
            for example in self.examples:
                parts.append(f"- {example}")
            parts.append("")
        
        # Add constraints if available
        if self.constraints:
            parts.append("Constraints:")
            for constraint in self.constraints:
                parts.append(f"- {constraint}")
            parts.append("")
        
        # Add the main instruction
        parts.append(f"Instruction:\n{self.instruction}")
        
        return "\n".join(parts)
