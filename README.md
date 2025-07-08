# Common NLP Tasks with Transformers

This project contains working examples for Chapter 06 of the Hugging Face Transformers book.

## Overview

Learn how to implement and understand:

## Prerequisites

- Python 3.12 (managed via pyenv)
- Poetry for dependency management
- Go Task for build automation
- API keys for any required services (see .env.example)

## Setup

1. Clone this repository
2. Run the setup task:
   ```bash
   task setup
   ```
3. Copy `.env.example` to `.env` and configure as needed

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and utilities
│   ├── main.py                # Entry point with all examples
│   ├── named_entity_recognition.py        # Named Entity Recognition implementation
│   ├── question_answering.py        # Question Answering implementation
│   ├── text_generation.py        # Text Generation implementation
│   ├── multi_task_learning.py        # Multi Task Learning implementation
│   └── utils.py               # Utility functions
├── tests/
│   └── test_examples.py       # Unit tests
├── .env.example               # Environment template
├── Taskfile.yml               # Task automation
└── pyproject.toml             # Poetry configuration
```

## Running Examples

Run key examples (recommended for quick start):
```bash
task run-simple
```

Run all examples (may take longer):
```bash
task run
```

Or run individual modules:
```bash
task run-named-entity-recognition    # Run named entity recognition
task run-question-answering    # Run question answering
task run-text-generation    # Run text generation
```

## Interactive Jupyter Notebook

For the best learning experience, use the interactive notebook:
```bash
task notebook    # Launch Jupyter Notebook
# or
task lab        # Launch Jupyter Lab
```

The notebook includes:
- All examples with visualizations
- Step-by-step explanations
- Interactive code you can modify
- Performance analytics and security demonstrations

## Available Tasks

- `task setup` - Set up Python environment and install dependencies
- `task run` - Run all examples
- `task test` - Run unit tests
- `task format` - Format code with Black and Ruff
- `task clean` - Clean up generated files

## Learn More

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Library](https://github.com/huggingface/transformers)
- [Book Resources](https://example.com/book-resources)
