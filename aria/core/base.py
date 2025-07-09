# aria/core/base.py
"""
Базовые классы и протоколы для ARIA архитектуры
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
import threading
import time
import logging

from .types import (
    ModalityType, ProcessingStage, ModalityData, 
    ProcessingResult, ComponentInfo, ComponentError
)

# Настройка логирования
logger = logging.getLogger(__name__)


@runtime_checkable
class Tokenizer(Protocol):
    """Протокол для токенизаторов"""
    
    def tokenize(self, data: ModalityData) -> List[str]:
        """Токенизирует данные в список токенов"""
        ...
    
    def detokenize(self, tokens: List[str]) -> ModalityData:
        """Преобразует токены обратно в данные"""
        ...


@runtime_checkable  
class Encoder(Protocol):
    """Протокол для энкодеров"""
    
    def encode(self, tokens: List[str]) -> Any:
        """Кодирует токены в векторное представление"""
        ...
    
    def decode(self, embedding: Any) -> List[str]:
        """Декодирует векторное представление в токены"""
        ...


@runtime_checkable
class FusionStrategy(Protocol):
    """Протокол для стратегий слияния модальностей"""
    
    def fuse(self, inputs: List[ProcessingResult]) -> ProcessingResult:
        """Объединяет результаты из разных модальностей"""
        ...


@runtime_checkable
class Generator(Protocol):
    """Протокол для генераторов"""
    
    def generate(self, context: ProcessingResult, 
                max_length: int = 50) -> ProcessingResult:
        """Генерирует новые данные на основе контекста"""
        ...


class Component(ABC):
    """
    Базовый компонент системы ARIA
    
    Все компоненты должны наследоваться от этого класса
    и реализовывать метод process()
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Инициализация компонента
        
        Args:
            name: Уникальное имя компонента
            config: Конфигурация компонента
        """
        self.name = name
        self.config = config or {}
        self.metadata = {}
        self._lock = threading.Lock()
        self._stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'last_call_time': None,
            'errors': 0
        }
        
        # Валидация конфигурации
        self._validate_config()
        
        logger.debug(f"Инициализирован компонент {self.name} типа {self.__class__.__name__}")
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Основная функция обработки данных
        
        Args:
            input_data: Входные данные для обработки
            
        Returns:
            Результат обработки
            
        Raises:
            ComponentError: При ошибке обработки
        """
        pass
    
    def _validate_config(self) -> None:
        """Валидация конфигурации компонента"""
        # Базовая валидация - может быть переопределена в наследниках
        if not isinstance(self.config, dict):
            raise ComponentError(f"Конфигурация компонента {self.name} должна быть словарем")
    
    def safe_process(self, input_data: Any) -> ProcessingResult:
        """
        Безопасная обработка с обработкой ошибок и статистикой
        
        Args:
            input_data: Входные данные
            
        Returns:
            ProcessingResult с результатом или ошибкой
        """
        start_time = time.time()
        
        try:
            with self._lock:
                self._stats['total_calls'] += 1
                self._stats['last_call_time'] = start_time
            
            # Выполняем обработку
            result = self.process(input_data)
            
            # Создаем ProcessingResult если компонент вернул сырые данные
            if not isinstance(result, ProcessingResult):
                result = ProcessingResult(
                    data=result,
                    stage=self._get_default_stage(),
                    modality=self._infer_modality(result),
                    component_name=self.name
                )
            
            # Добавляем информацию о времени
            processing_time = time.time() - start_time
            result.add_timing_info(start_time, start_time + processing_time)
            result.component_name = self.name
            
            # Обновляем статистику
            with self._lock:
                self._stats['total_time'] += processing_time
            
            logger.debug(f"Компонент {self.name} успешно обработал данные за {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            # Обрабатываем ошибку
            processing_time = time.time() - start_time
            
            with self._lock:
                self._stats['errors'] += 1
                self._stats['total_time'] += processing_time
            
            error_msg = f"Ошибка в компоненте {self.name}: {str(e)}"
            logger.error(error_msg)
            
            return ProcessingResult(
                data=None,
                stage=self._get_default_stage(),
                modality=ModalityType.TEXT,  # Fallback
                error=error_msg,
                component_name=self.name,
                processing_time=processing_time
            )
    
    def _get_default_stage(self) -> ProcessingStage:
        """Возвращает стадию по умолчанию для компонента"""
        # Можно переопределить в наследниках
        return ProcessingStage.PREPROCESSING
    
    def _infer_modality(self, data: Any) -> ModalityType:
        """Пытается определить модальность данных"""
        if isinstance(data, str):
            return ModalityType.TEXT
        elif isinstance(data, list) and data and isinstance(data[0], str):
            return ModalityType.TEXT
        elif hasattr(data, 'shape') and len(data.shape) >= 2:
            return ModalityType.IMAGE  # Предполагаем изображение для многомерных массивов
        else:
            return ModalityType.NUMERIC
    
    def get_info(self) -> ComponentInfo:
        """Возвращает информацию о компоненте"""
        return ComponentInfo(
            name=self.name,
            type=self.__class__.__name__,
            description=self.__doc__ or "",
            parameters=self.config,
            supported_modalities=getattr(self, 'supported_modalities', [])
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы компонента"""
        with self._lock:
            stats = self._stats.copy()
        
        if stats['total_calls'] > 0:
            stats['avg_time'] = stats['total_time'] / stats['total_calls']
            stats['error_rate'] = stats['errors'] / stats['total_calls']
        else:
            stats['avg_time'] = 0.0
            stats['error_rate'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику"""
        with self._lock:
            self._stats = {
                'total_calls': 0,
                'total_time': 0.0,
                'last_call_time': None,
                'errors': 0
            }
    
    def configure(self, new_config: Dict[str, Any]) -> None:
        """Обновляет конфигурацию компонента"""
        old_config = self.config.copy()
        self.config.update(new_config)
        
        try:
            self._validate_config()
            self._on_config_changed(old_config, self.config)
        except Exception as e:
            # Откатываем изменения при ошибке
            self.config = old_config
            raise ComponentError(f"Ошибка обновления конфигурации {self.name}: {e}")
    
    def _on_config_changed(self, old_config: Dict[str, Any], 
                          new_config: Dict[str, Any]) -> None:
        """Callback при изменении конфигурации"""
        # Может быть переопределен в наследниках
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"


class MultiModalComponent(Component):
    """
    Базовый класс для мультимодальных компонентов
    
    Поддерживает обработку нескольких типов модальностей
    """
    
    def __init__(self, name: str, supported_modalities: List[ModalityType], 
                 config: Dict[str, Any] = None):
        """
        Инициализация мультимодального компонента
        
        Args:
            name: Имя компонента
            supported_modalities: Список поддерживаемых модальностей
            config: Конфигурация
        """
        super().__init__(name, config)
        self.supported_modalities = supported_modalities
        
        if not supported_modalities:
            raise ComponentError(f"Компонент {name} должен поддерживать хотя бы одну модальность")
        
        logger.debug(f"Мультимодальный компонент {name} поддерживает: {[m.value for m in supported_modalities]}")
    
    def supports_modality(self, modality: ModalityType) -> bool:
        """Проверяет поддержку модальности"""
        return modality in self.supported_modalities
    
    def validate_input(self, data: ModalityData) -> bool:
        """Проверяет совместимость входных данных"""
        if not isinstance(data, ModalityData):
            return False
        return self.supports_modality(data.modality)
    
    def _get_default_stage(self) -> ProcessingStage:
        """Мультимодальные компоненты по умолчанию выполняют слияние"""
        return ProcessingStage.FUSION
    
    def _infer_modality(self, data: Any) -> ModalityType:
        """Для мультимодальных компонентов результат может быть мультимодальным"""
        if len(self.supported_modalities) == 1:
            return self.supported_modalities[0]
        else:
            return ModalityType.MULTIMODAL


class StatefulComponent(Component):
    """
    Компонент с состоянием
    
    Может накапливать информацию между вызовами
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self._state = {}
        self._state_lock = threading.RLock()  # Reentrant lock
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Получить значение из состояния"""
        with self._state_lock:
            return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Установить значение в состоянии"""
        with self._state_lock:
            self._state[key] = value
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Обновить несколько значений состояния"""
        with self._state_lock:
            self._state.update(updates)
    
    def clear_state(self) -> None:
        """Очистить состояние"""
        with self._state_lock:
            self._state.clear()
    
    def get_full_state(self) -> Dict[str, Any]:
        """Получить полное состояние (копию)"""
        with self._state_lock:
            return self._state.copy()


class BatchComponent(Component):
    """
    Компонент для батчевой обработки
    
    Может эффективно обрабатывать множество элементов сразу
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.batch_size = config.get('batch_size', 32)
        self.enable_batching = config.get('enable_batching', True)
    
    @abstractmethod
    def process_batch(self, batch: List[Any]) -> List[Any]:
        """
        Обрабатывает батч данных
        
        Args:
            batch: Список входных данных
            
        Returns:
            Список результатов (того же размера что и входной батч)
        """
        pass
    
    def process(self, input_data: Any) -> Any:
        """
        Обработка одного элемента (может использовать батчинг внутри)
        
        По умолчанию просто вызывает process_batch с одним элементом
        """
        if self.enable_batching:
            results = self.process_batch([input_data])
            return results[0] if results else None
        else:
            return self.process_single(input_data)
    
    def process_single(self, input_data: Any) -> Any:
        """
        Обработка одного элемента без батчинга
        
        Может быть переопределена для более эффективной обработки
        """
        results = self.process_batch([input_data])
        return results[0] if results else None
    
    def process_multiple(self, input_list: List[Any]) -> List[Any]:
        """
        Обработка списка элементов с автоматическим батчингом
        """
        if not input_list:
            return []
        
        if not self.enable_batching or len(input_list) <= self.batch_size:
            return self.process_batch(input_list)
        
        # Разбиваем на батчи
        results = []
        for i in range(0, len(input_list), self.batch_size):
            batch = input_list[i:i + self.batch_size]
            batch_results = self.process_batch(batch)
            results.extend(batch_results)
        
        return results