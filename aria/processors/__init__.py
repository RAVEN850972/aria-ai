# aria/processors/__init__.py
"""
ARIA Processors Module

Модуль обработки данных для различных модальностей:
- Text: обработка текста, токенизация, n-граммы, генерация
- Image: обработка изображений, патчи, признаки
- Audio: обработка аудио, спектральный анализ
- Video: обработка видео, временные последовательности
- Multimodal: кросс-модальная обработка и слияние
"""

from typing import Dict, Any, List
import logging

# Импорты из подмодулей будут добавлены по мере их создания
from .text import *

logger = logging.getLogger(__name__)

# Версия модуля процессоров
__version__ = "0.1.0"

# Метаданные доступных процессоров
PROCESSOR_METADATA = {
    "text": {
        "description": "Обработка текстовых данных",
        "components": [
            "AdvancedTextTokenizer",
            "NGramExtractor", 
            "BeamSearchGenerator",
            "TextFusionStrategy",
            "TextModelManager"
        ],
        "modalities": ["text"],
        "dependencies": []
    }
    # Другие модальности будут добавлены позже
}

def get_available_processors() -> Dict[str, Any]:
    """
    Возвращает информацию о доступных процессорах
    
    Returns:
        Словарь с метаданными процессоров
    """
    return PROCESSOR_METADATA.copy()

def get_processors_by_modality(modality: str) -> List[str]:
    """
    Возвращает список процессоров для указанной модальности
    
    Args:
        modality: Тип модальности
        
    Returns:
        Список имен процессоров
    """
    result = []
    for processor_type, metadata in PROCESSOR_METADATA.items():
        if modality in metadata["modalities"]:
            result.extend(metadata["components"])
    return result

def register_all_processors():
    """
    Регистрирует все доступные процессоры в глобальном реестре
    """
    from ..core import GLOBAL_REGISTRY
    
    # Проверяем, не зарегистрированы ли уже компоненты
    if len(GLOBAL_REGISTRY.list_types()) > 0:
        logger.debug("Процессоры уже зарегистрированы, пропускаем")
        return
    
    logger.info("Регистрация всех процессоров...")
    
    # Регистрируем текстовые процессоры
    try:
        from .text import register_text_processors
        register_text_processors()
        logger.info("✅ Текстовые процессоры зарегистрированы")
    except ImportError as e:
        logger.warning(f"⚠️ Не удалось загрузить текстовые процессоры: {e}")
    
    total_registered = len(GLOBAL_REGISTRY.list_types())
    logger.info(f"🎉 Всего зарегистрировано процессоров: {total_registered}")

def validate_processor_dependencies() -> Dict[str, List[str]]:
    """
    Проверяет зависимости всех процессоров
    
    Returns:
        Словарь с нарушениями зависимостей
    """
    from ..core import GLOBAL_REGISTRY
    return GLOBAL_REGISTRY.validate_all_dependencies()

__all__ = [
    # Функции управления
    "get_available_processors",
    "get_processors_by_modality", 
    "register_all_processors",
    "validate_processor_dependencies",
    
    # Метаданные
    "PROCESSOR_METADATA",
    "__version__",
    
    # Экспорты из подмодулей (будут расширены)
]

# Автоматическая регистрация при импорте модуля
logger.debug("Инициализация модуля processors")