# aria/processors/text/__init__.py
"""
ARIA Text Processing Module

Модуль обработки текстовых данных, включающий:
- Продвинутую токенизацию с сохранением контекста
- Извлечение переменных n-грамм с адаптивными весами
- Генерацию текста с beam search и nucleus sampling
- Стратегии слияния текстовых данных
- Управление моделями и сохранение состояния
"""

from .tokenizers import AdvancedTextTokenizer
from .ngrams import NGramExtractor, NGramConfig, BeamCandidate
from .generators import BeamSearchGenerator, GenerationConfig
from .fusion import TextFusionStrategy, ModelManager

__all__ = [
    # Tokenizers
    "AdvancedTextTokenizer",
    
    # N-grams
    "NGramExtractor",
    "NGramConfig", 
    "BeamCandidate",
    
    # Generators
    "BeamSearchGenerator",
    "GenerationConfig",
    
    # Fusion & Management
    "TextFusionStrategy",
    "ModelManager",
    
    # Utility function
    "register_text_processors"
]

def register_text_processors():
    """Регистрирует все текстовые процессоры в глобальном реестре"""
    from ...core import GLOBAL_REGISTRY
    
    # Регистрируем компоненты с проверкой на существование
    components = [
        ("AdvancedTextTokenizer", AdvancedTextTokenizer),
        ("NGramExtractor", NGramExtractor),
        ("BeamSearchGenerator", BeamSearchGenerator),
        ("TextFusionStrategy", TextFusionStrategy),
        ("ModelManager", ModelManager)
    ]
    
    for name, component_class in components:
        if name not in GLOBAL_REGISTRY:
            GLOBAL_REGISTRY.register(name, component_class)

__version__ = "0.1.0"