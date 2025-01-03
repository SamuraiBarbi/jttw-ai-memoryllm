# JTTW-AI MemoryLLM: An Autonomous Prompt Enhancement System

## Overview
This project implements an intelligent prompt optimization system that leverages DSPy's language model programming (LMP) framework and PydanticAI's robust data validation to create a self-improving prompt enhancement pipeline. By combining DSPy's teleprompter and compiler capabilities with Pydantic's type safety and validation, we create a system that automatically evolves and refines prompts for optimal LLM performance.

## Key Technologies

### DSPy Integration
- **Advanced Optimizers**: 
  - MIPRO (Mixed Initiative PROmpting) for interactive prompt exploration
  - COPRO (COntextual PROmpting) for context-aware refinement
  - Bayesian optimization for example-instruction pairing
- **Teleprompter**: Leverages DSPy's signature feature for automatic prompt optimization
- **Bootstrapped Fine-tuning**: Task-specific optimization through iterative learning
- **Few-Shot Generation**: Automatic generation of relevant examples for enhanced performance
- **Chain-of-Thought Programming**: Implements complex reasoning chains using DSPy's modular components
- **Compiled Prompts**: Utilizes DSPy's prompt compilation for optimized execution
- **Forward-Backward Signatures**: Employs DSPy's signature system for robust prompt validation

### PydanticAI Capabilities
- **Type-Safe Prompt Templates**: Ensures prompt structure consistency with Pydantic models
- **Validated Response Parsing**: Automatically validates and structures LLM outputs
- **Schema Evolution**: Tracks and manages prompt schema changes over optimization iterations
- **Custom Field Validators**: Implements domain-specific validation rules for prompt components
- **JSON Schema Generation**: Automatic documentation of prompt structures

## Core Features
- **Autonomous Enhancement**: 
  - Self-improving prompts using DSPy's teleprompter
  - MIPRO/COPRO optimization strategies
  - Bayesian optimization of example-instruction pairs
  - Automatic validation and correction via Pydantic models
  - Continuous learning from execution metrics

- **Iterative Optimization**: 
  - DSPy-powered optimization loops
  - Bootstrapped fine-tuning for task specialization
  - Automatic few-shot example generation
  - Bayesian search over example-instruction space
  - Progressive prompt refinement with validated improvements

- **Quality Assurance**:
  - Pydantic model validation for output consistency
  - DSPy metric evaluation framework
  - Example-based performance validation
  - Bayesian performance tracking
  - Structured error handling and recovery

## Technical Architecture
1. **Prompt Definition Layer**
   - Pydantic models define prompt structure
   - DSPy signatures specify expected behaviors
   - Few-shot example templates
   - Instruction-example pairing schemas
   - Type hints ensure consistency

2. **Optimization Pipeline**
   - MIPRO/COPRO optimizers for prompt refinement
   - Bootstrapped fine-tuning modules
   - Bayesian optimization engine
   - Metric-guided learning modules
   - Example generation engine

3. **Validation Framework**
   - Pydantic-enforced schema validation
   - DSPy forward-backward checking
   - Few-shot performance validation
   - Bayesian performance monitoring
   - Runtime type safety

4. **Execution Engine**
   - DSPy compiled prompt execution
   - Dynamic example selection
   - Instruction-example pair optimization
   - Structured response parsing
   - Performance metric collection

## Development Roadmap

### Phase 1: Enhanced Optimization (Q1 2025)
✅ Implement MIPRO (Mixed Initiative PROmpting)
  - [x] Interactive prompt exploration with multiple optimization strategies
  - [x] User feedback integration through analysis history
  - [x] Dynamic prompt adjustment via semantic, structural, and contextual refinement
  - [x] Advanced refinement strategies (perspective, challenge, analogy, misconception)

✅ Develop COPRO (COntextual PROmpting)
  - [x] Context-aware refinement with analysis system
  - [x] Semantic understanding through context processing
  - [x] Basic adaptive prompt generation
  - [x] Advanced context analysis and variation generation
  - [x] Multi-audience adaptation strategies
  - [x] Interactive learning elements

✅ Implement Hybrid Optimization
  - [x] MIPRO-COPRO integration
  - [x] Sequential optimization pipeline
  - [x] Combined performance tracking
  - [x] Advanced strategy selection
  - [ ] Dynamic switching between optimizers (In Progress)

🚧 Add Bayesian Optimization
  - [ ] Example-instruction pair optimization (Not Started)
  - [ ] Performance metric tracking with Bayesian approach (Not Started)
  - [ ] Hyperparameter tuning system (Not Started)

**Current Progress (January 2025)**:
- Overall Phase 1 Completion: ~85%
- Core MIPRO and COPRO frameworks fully implemented with advanced features
- Hybrid optimization strategy enhanced with sophisticated refinement methods
- Chain of thought reasoning integrated with multiple perspective analysis
- Performance tracking system in place with detailed context scoring
- Multi-strategy optimization working with audience adaptation

**Remaining Tasks**:
1. Implement full Bayesian optimization module
2. Complete dynamic optimizer switching system
3. Implement comprehensive testing suite
4. Add proper logging and error handling
5. Fine-tune hybrid strategy selection

**Technical Improvements Needed**:
- Validation for LLM responses
- API documentation
- Performance benchmarking
- Error recovery mechanisms
- Hybrid strategy selection optimization

On track for Q1 2025 completion with primary focus on Bayesian optimization implementation and dynamic optimizer switching.

### Phase 2: DSPy Core Integration (Q2 2025)
- [ ] Teleprompter Implementation
  - Automatic prompt optimization
  - Performance monitoring
  - Feedback loop integration
- [ ] Few-Shot Learning System
  - Example generation engine
  - Dynamic example selection
  - Performance validation
- [ ] Bootstrapped Fine-tuning
  - Task-specific optimization
  - Iterative learning
  - Performance tracking

### Phase 3: Advanced Validation (Q3 2025)
- [ ] Enhanced PydanticAI Features
  - Schema evolution tracking
  - Custom field validators
  - JSON schema generation
- [ ] DSPy Forward-Backward System
  - Signature validation
  - Runtime type safety
  - Error handling improvements
- [ ] Performance Validation Framework
  - Metric collection
  - Quality assurance
  - Automated testing

### Phase 4: Pipeline Optimization (Q4 2025)
- [ ] Example Management System
  - Automated generation
  - Quality assessment
  - Storage and retrieval
- [ ] Metric-Guided Learning
  - Performance metrics
  - Learning rate adaptation
  - Optimization strategies
- [ ] Production Readiness
  - System monitoring
  - Error recovery
  - Performance optimization

### Current Status (January 2025)
✅ Implemented:
- Chain-of-Thought reasoning
- Basic DSPy integration
- Pydantic model validation
- Initial prompt optimization pipeline

🚧 In Progress:
- Advanced optimization strategies
- Example generation system
- Performance monitoring

Each phase builds upon the previous ones, with a focus on maintaining stability while adding new features. The roadmap is subject to adjustment based on user feedback and emerging requirements.

## Implementation Example

Here are some code examples demonstrating the usage of the system:

```python
# Example of defining a DSPy signature
from dsp import Signature

signature = Signature(...)

# Example of configuring the MIPRO optimizer
from optimization_pipeline import MIPRO

mipro_optimizer = MIPRO(...)
```

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Conda package manager
- Ollama with `llama3.2:3b` model installed

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SamuraiBarbi/jttw-ai-memoryllm.git
cd jttw-ai-memoryllm
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate jttw-ai-memoryllm
```

### Testing

1. Make sure Ollama is running with the llama3.2:3b model:


2. Run the main script to test the prompt optimization:
```bash
python main.py
```

The script will:
- Load an initial prompt about neural networks
- Run through multiple iterations of optimization with chain of thought reasoning
- Generate an optimized prompt
- Create a well-structured, beginner-friendly response

### Environment Details

The project uses a conda environment with the following main dependencies:
- dspy-ai: For language model programming and prompt optimization
- litellm: For interfacing with the Ollama inference server
- pydantic: For data validation and structured prompt definitions
- numpy & pandas: For data manipulation and analysis

The full list of dependencies can be found in `environment.yml`.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
