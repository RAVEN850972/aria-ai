# aria/core/__init__.py
"""
ARIA Core Module

Основные компоненты архитектуры ARIA:
- Базовые классы и протоколы
- Граф обработки данных
- Реестр компонентов
- Фабрика графов
- Типы данных
"""

from .types import (
    ModalityType,
    ProcessingStage,
    ModalityData,
    ProcessingResult
)

from .base import (
    Component,
    MultiModalComponent,
    StatefulComponent,
    BatchComponent,
    Tokenizer,
    Encoder,
    FusionStrategy,
    Generator
)

from .graph import (
    GraphNode,
    ProcessingGraph
)

from .registry import (
    ComponentRegistry,
    GLOBAL_REGISTRY
)

from .factory import (
    GraphFactory,
    ComponentConfig,
    PipelineConfig
)

__all__ = [
    # Types
    'ModalityType',
    'ProcessingStage', 
    'ModalityData',
    'ProcessingResult',
    
    # Base classes
    'Component',
    'MultiModalComponent',
    'StatefulComponent',
    'BatchComponent',
    'Tokenizer',
    'Encoder',
    'FusionStrategy',
    'Generator',
    
    # Graph system
    'GraphNode',
    'ProcessingGraph',
    
    # Registry
    'ComponentRegistry',
    'GLOBAL_REGISTRY',
    
    # Factory
    'GraphFactory',
    'ComponentConfig',
    'PipelineConfig'
]

__version__ = "0.1.0"
__author__ = "ARIA Team"