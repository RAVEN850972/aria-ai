# aria/__init__.py
"""
ARIA - Adaptive Reconfigurable Intelligence Architecture

Модульная система для мультимодальной обработки данных с графовой архитектурой.
"""

# Импорты основных модулей
from . import core
from . import processors
from . import dsl

# Основные классы и функции
from .core import (
    GLOBAL_REGISTRY,
    ProcessingGraph,
    Component,
    ModalityType,
    ProcessingResult
)

from .processors import register_all_processors

from .dsl import (
    FluentPipelineBuilder,
    PipelineDSL, 
    TemplateManager,
    create_text_pipeline
)

# Версия
__version__ = "0.1.0"
__author__ = "ARIA Team"

# Экспорты
__all__ = [
    # Core
    'GLOBAL_REGISTRY',
    'ProcessingGraph', 
    'Component',
    'ModalityType',
    'ProcessingResult',
    
    # Processors
    'register_all_processors',
    
    # DSL
    'FluentPipelineBuilder',
    'PipelineDSL',
    'TemplateManager', 
    'create_text_pipeline',
    
    # Modules
    'core',
    'processors',
    'dsl',
]

# Автоматическая инициализация
def quick_setup():
    """Быстрая настройка ARIA для немедленного использования"""
    register_all_processors()
    return GLOBAL_REGISTRY

# Ленивая инициализация
_initialized = False

def ensure_initialized():
    """Гарантирует инициализацию системы"""
    global _initialized
    if not _initialized:
        register_all_processors()
        _initialized = True