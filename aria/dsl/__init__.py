# aria/dsl/__init__.py
"""ARIA DSL Module"""

from .parser import PipelineDSL
from .fluent import FluentPipelineBuilder
from .templates import TemplateManager, get_builtin_templates, create_text_pipeline
from .cli import PipelineCLI

__all__ = [
    'PipelineDSL',
    'FluentPipelineBuilder', 
    'TemplateManager',
    'get_builtin_templates',
    'create_text_pipeline',
    'PipelineCLI'
]