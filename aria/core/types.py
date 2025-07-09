# aria/core/types.py
"""
Основные типы данных для ARIA архитектуры
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import time


class ModalityType(Enum):
    """Типы поддерживаемых модальностей"""
    TEXT = "text"
    IMAGE = "image" 
    AUDIO = "audio"
    VIDEO = "video"
    NUMERIC = "numeric"
    GRAPH = "graph"
    MULTIMODAL = "multimodal"


class ProcessingStage(Enum):
    """Этапы обработки данных"""
    TOKENIZATION = "tokenization"
    ENCODING = "encoding"
    FUSION = "fusion"
    GENERATION = "generation"
    DECODING = "decoding"
    ANALYSIS = "analysis"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"


@dataclass
class ModalityData:
    """
    Универсальный контейнер для данных любой модальности
    
    Args:
        data: Данные (текст, изображение, аудио и т.д.)
        modality: Тип модальности
        metadata: Дополнительные метаданные
        timestamp: Время создания
        source: Источник данных
        encoding: Кодировка для текстовых данных
    """
    data: Any
    modality: ModalityType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    encoding: Optional[str] = None
    
    def __post_init__(self):
        """Валидация после создания"""
        if not isinstance(self.modality, ModalityType):
            if isinstance(self.modality, str):
                try:
                    self.modality = ModalityType(self.modality)
                except ValueError:
                    raise ValueError(f"Неизвестный тип модальности: {self.modality}")
            else:
                raise TypeError("modality должен быть ModalityType или строкой")
    
    def get_size(self) -> int:
        """Возвращает размер данных"""
        if hasattr(self.data, '__len__'):
            return len(self.data)
        elif hasattr(self.data, 'shape'):
            return self.data.shape[0] if self.data.shape else 0
        else:
            return 1
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Получить значение из метаданных"""
        return self.metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Установить значение в метаданных"""
        self.metadata[key] = value
    
    def copy(self) -> 'ModalityData':
        """Создать копию с теми же данными"""
        return ModalityData(
            data=self.data,
            modality=self.modality,
            metadata=self.metadata.copy(),
            timestamp=self.timestamp,
            source=self.source,
            encoding=self.encoding
        )


@dataclass
class ProcessingResult:
    """
    Результат обработки на любом этапе пайплайна
    
    Args:
        data: Результирующие данные
        stage: Этап обработки
        modality: Тип модальности результата
        score: Оценка качества/уверенности (0.0-1.0)
        metadata: Метаданные обработки
        processing_time: Время обработки в секундах
        error: Описание ошибки, если есть
        component_name: Имя компонента, создавшего результат
    """
    data: Any
    stage: ProcessingStage
    modality: ModalityType
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    error: Optional[str] = None
    component_name: Optional[str] = None
    
    def __post_init__(self):
        """Валидация после создания"""
        if not isinstance(self.stage, ProcessingStage):
            if isinstance(self.stage, str):
                try:
                    self.stage = ProcessingStage(self.stage)
                except ValueError:
                    raise ValueError(f"Неизвестный этап обработки: {self.stage}")
            else:
                raise TypeError("stage должен быть ProcessingStage или строкой")
        
        if not isinstance(self.modality, ModalityType):
            if isinstance(self.modality, str):
                try:
                    self.modality = ModalityType(self.modality)
                except ValueError:
                    raise ValueError(f"Неизвестный тип модальности: {self.modality}")
            else:
                raise TypeError("modality должен быть ModalityType или строкой")
        
        # Нормализуем score
        self.score = max(0.0, min(1.0, float(self.score)))
    
    def is_successful(self) -> bool:
        """Проверяет успешность обработки"""
        return self.error is None
    
    def get_data_size(self) -> int:
        """Возвращает размер результирующих данных"""
        if hasattr(self.data, '__len__'):
            return len(self.data)
        elif hasattr(self.data, 'shape'):
            return self.data.shape[0] if self.data.shape else 0
        else:
            return 1
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Получить значение из метаданных"""
        return self.metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Установить значение в метаданных"""
        self.metadata[key] = value
    
    def add_timing_info(self, start_time: float, end_time: Optional[float] = None) -> None:
        """Добавляет информацию о времени обработки"""
        if end_time is None:
            end_time = time.time()
        self.processing_time = end_time - start_time
        self.metadata['start_time'] = start_time
        self.metadata['end_time'] = end_time
    
    def to_modality_data(self) -> ModalityData:
        """Преобразует в ModalityData для дальнейшей обработки"""
        return ModalityData(
            data=self.data,
            modality=self.modality,
            metadata=self.metadata.copy(),
            source=self.component_name
        )
    
    def __str__(self) -> str:
        status = "Success" if self.is_successful() else f"Error: {self.error}"
        return (f"ProcessingResult({self.stage.value}, {self.modality.value}, "
                f"score={self.score:.3f}, time={self.processing_time:.3f}s, {status})")


@dataclass
class ComponentInfo:
    """Информация о компоненте"""
    name: str
    type: str
    version: str = "1.0.0"
    description: str = ""
    supported_modalities: list = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь"""
        return {
            'name': self.name,
            'type': self.type,
            'version': self.version,
            'description': self.description,
            'supported_modalities': [m.value if isinstance(m, ModalityType) else m 
                                   for m in self.supported_modalities],
            'parameters': self.parameters,
            'dependencies': self.dependencies
        }


# Константы для стандартных значений
DEFAULT_SCORE = 0.0
DEFAULT_PROCESSING_TIME = 0.0
UNKNOWN_COMPONENT = "unknown"

# Исключения для типизированных ошибок
class ARIAException(Exception):
    """Базовое исключение ARIA"""
    pass


class ModalityError(ARIAException):
    """Ошибка связанная с модальностью"""
    pass


class ProcessingError(ARIAException):
    """Ошибка обработки данных"""
    pass


class ComponentError(ARIAException):
    """Ошибка компонента"""
    pass


class ConfigurationError(ARIAException):
    """Ошибка конфигурации"""
    pass