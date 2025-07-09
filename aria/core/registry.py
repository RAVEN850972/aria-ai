# aria/core/registry.py
"""
Система реестра компонентов для ARIA архитектуры
"""

from typing import Dict, Type, Optional, List, Any, Callable
import threading
import inspect
import logging
from collections import defaultdict

from .base import Component
from .types import ComponentInfo, ComponentError

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Реестр компонентов для динамического создания и управления
    
    Обеспечивает:
    - Регистрацию типов компонентов
    - Создание экземпляров с валидацией
    - Управление зависимостями
    - Версионирование компонентов
    - Поиск и фильтрацию
    """
    
    def __init__(self):
        """Инициализация реестра"""
        self._components: Dict[str, Type[Component]] = {}
        self._instances: Dict[str, Component] = {}
        self._metadata: Dict[str, ComponentInfo] = {}
        self._dependencies: Dict[str, List[str]] = defaultdict(list)
        self._creation_hooks: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._stats = {
            'total_registrations': 0,
            'total_creations': 0,
            'creation_errors': 0
        }
        
        logger.debug("Инициализирован реестр компонентов")
    
    def register(self, name: str, component_class: Type[Component], 
                 metadata: Optional[ComponentInfo] = None,
                 dependencies: Optional[List[str]] = None,
                 force: bool = False) -> None:
        """
        Регистрирует класс компонента
        
        Args:
            name: Уникальное имя типа компонента
            component_class: Класс компонента
            metadata: Метаданные компонента
            dependencies: Список зависимостей
            force: Принудительная перерегистрация
        """
        with self._lock:
            # Проверяем существование
            if name in self._components and not force:
                raise ComponentError(f"Компонент '{name}' уже зарегистрирован")
            
            # Валидация класса
            if not inspect.isclass(component_class):
                raise ComponentError(f"'{component_class}' не является классом")
            
            if not issubclass(component_class, Component):
                raise ComponentError(f"Класс '{component_class.__name__}' должен наследоваться от Component")
            
            # Создаем метаданные если не предоставлены
            if metadata is None:
                metadata = self._create_metadata_from_class(name, component_class)
            
            # Регистрируем
            self._components[name] = component_class
            self._metadata[name] = metadata
            
            if dependencies:
                self._dependencies[name] = dependencies.copy()
            
            self._stats['total_registrations'] += 1
            
            logger.info(f"Зарегистрирован компонент '{name}': {component_class.__name__}")
    
    def unregister(self, name: str, remove_instances: bool = True) -> None:
        """
        Отменяет регистрацию компонента
        
        Args:
            name: Имя компонента
            remove_instances: Удалить все экземпляры
        """
        with self._lock:
            if name not in self._components:
                raise ComponentError(f"Компонент '{name}' не зарегистрирован")
            
            # Удаляем экземпляры если требуется
            if remove_instances:
                instances_to_remove = [inst_name for inst_name, inst in self._instances.items()
                                     if inst.__class__ == self._components[name]]
                for inst_name in instances_to_remove:
                    del self._instances[inst_name]
            
            # Удаляем регистрацию
            del self._components[name]
            del self._metadata[name]
            
            if name in self._dependencies:
                del self._dependencies[name]
            
            if name in self._creation_hooks:
                del self._creation_hooks[name]
            
            logger.info(f"Отменена регистрация компонента '{name}'")
    
    def create(self, instance_name: str, component_type: str, 
               config: Dict[str, Any] = None,
               validate_dependencies: bool = True) -> Component:
        """
        Создает экземпляр компонента
        
        Args:
            instance_name: Уникальное имя экземпляра
            component_type: Тип компонента
            config: Конфигурация
            validate_dependencies: Проверять зависимости
            
        Returns:
            Созданный экземпляр компонента
        """
        with self._lock:
            # Проверяем тип
            if component_type not in self._components:
                available = list(self._components.keys())
                raise ComponentError(f"Неизвестный тип компонента '{component_type}'. "
                                   f"Доступные типы: {available}")
            
            # Проверяем уникальность имени
            if instance_name in self._instances:
                raise ComponentError(f"Экземпляр с именем '{instance_name}' уже существует")
            
            # Проверяем зависимости
            if validate_dependencies:
                self._validate_dependencies(component_type)
            
            try:
                # Создаем экземпляр
                component_class = self._components[component_type]
                config = config or {}
                
                # Выполняем pre-creation хуки
                self._execute_creation_hooks(component_type, 'pre', instance_name, config)
                
                instance = component_class(instance_name, config)
                
                # Сохраняем экземпляр
                self._instances[instance_name] = instance
                
                # Выполняем post-creation хуки
                self._execute_creation_hooks(component_type, 'post', instance_name, config)
                
                self._stats['total_creations'] += 1
                
                logger.debug(f"Создан экземпляр '{instance_name}' типа '{component_type}'")
                
                return instance
                
            except Exception as e:
                self._stats['creation_errors'] += 1
                error_msg = f"Ошибка создания экземпляра '{instance_name}' типа '{component_type}': {e}"
                logger.error(error_msg)
                raise ComponentError(error_msg) from e
    
    def get(self, instance_name: str) -> Optional[Component]:
        """
        Получает экземпляр компонента по имени
        
        Args:
            instance_name: Имя экземпляра
            
        Returns:
            Экземпляр компонента или None
        """
        with self._lock:
            return self._instances.get(instance_name)
    
    def get_or_create(self, instance_name: str, component_type: str,
                      config: Dict[str, Any] = None) -> Component:
        """
        Получает существующий экземпляр или создает новый
        
        Args:
            instance_name: Имя экземпляра
            component_type: Тип компонента
            config: Конфигурация для создания
            
        Returns:
            Экземпляр компонента
        """
        instance = self.get(instance_name)
        if instance is None:
            instance = self.create(instance_name, component_type, config)
        return instance
    
    def remove_instance(self, instance_name: str) -> bool:
        """
        Удаляет экземпляр компонента
        
        Args:
            instance_name: Имя экземпляра
            
        Returns:
            True если экземпляр был удален
        """
        with self._lock:
            if instance_name in self._instances:
                del self._instances[instance_name]
                logger.debug(f"Удален экземпляр '{instance_name}'")
                return True
            return False
    
    def list_types(self) -> List[str]:
        """Возвращает список зарегистрированных типов компонентов"""
        with self._lock:
            return list(self._components.keys())
    
    def list_instances(self) -> List[str]:
        """Возвращает список созданных экземпляров"""
        with self._lock:
            return list(self._instances.keys())
    
    def get_component_info(self, component_type: str) -> Optional[ComponentInfo]:
        """
        Получает информацию о типе компонента
        
        Args:
            component_type: Тип компонента
            
        Returns:
            Информация о компоненте или None
        """
        with self._lock:
            return self._metadata.get(component_type)
    
    def search_components(self, 
                         modality: Optional[str] = None,
                         description_keywords: Optional[List[str]] = None,
                         has_dependencies: Optional[bool] = None) -> List[str]:
        """
        Поиск компонентов по критериям
        
        Args:
            modality: Поддерживаемая модальность
            description_keywords: Ключевые слова в описании
            has_dependencies: Имеет ли зависимости
            
        Returns:
            Список подходящих типов компонентов
        """
        with self._lock:
            results = []
            
            for comp_type, metadata in self._metadata.items():
                # Фильтр по модальности
                if modality is not None:
                    supported = [m.value if hasattr(m, 'value') else str(m) 
                               for m in metadata.supported_modalities]
                    if modality not in supported:
                        continue
                
                # Фильтр по ключевым словам
                if description_keywords is not None:
                    description = metadata.description.lower()
                    if not any(keyword.lower() in description for keyword in description_keywords):
                        continue
                
                # Фильтр по зависимостям
                if has_dependencies is not None:
                    has_deps = bool(self._dependencies.get(comp_type))
                    if has_deps != has_dependencies:
                        continue
                
                results.append(comp_type)
            
            return results
    
    def get_dependencies(self, component_type: str) -> List[str]:
        """
        Получает список зависимостей компонента
        
        Args:
            component_type: Тип компонента
            
        Returns:
            Список зависимостей
        """
        with self._lock:
            return self._dependencies.get(component_type, []).copy()
    
    def add_creation_hook(self, component_type: str, hook: Callable,
                         phase: str = 'post') -> None:
        """
        Добавляет хук для создания компонента
        
        Args:
            component_type: Тип компонента
            hook: Функция-хук
            phase: Фаза выполнения ('pre' или 'post')
        """
        if phase not in ['pre', 'post']:
            raise ComponentError("Фаза хука должна быть 'pre' или 'post'")
        
        with self._lock:
            hook_key = f"{component_type}_{phase}"
            self._creation_hooks[hook_key].append(hook)
            
        logger.debug(f"Добавлен {phase}-хук для '{component_type}'")
    
    def get_instance_by_type(self, component_type: str) -> List[Component]:
        """
        Получает все экземпляры определенного типа
        
        Args:
            component_type: Тип компонента
            
        Returns:
            Список экземпляров
        """
        if component_type not in self._components:
            return []
        
        target_class = self._components[component_type]
        
        with self._lock:
            return [instance for instance in self._instances.values()
                   if isinstance(instance, target_class)]
    
    def validate_all_dependencies(self) -> Dict[str, List[str]]:
        """
        Проверяет все зависимости в реестре
        
        Returns:
            Словарь с нарушениями зависимостей
        """
        violations = {}
        
        with self._lock:
            for comp_type, deps in self._dependencies.items():
                missing_deps = []
                for dep in deps:
                    if dep not in self._components:
                        missing_deps.append(dep)
                
                if missing_deps:
                    violations[comp_type] = missing_deps
        
        return violations
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику реестра"""
        with self._lock:
            stats = self._stats.copy()
            stats.update({
                'registered_types': len(self._components),
                'active_instances': len(self._instances),
                'total_dependencies': sum(len(deps) for deps in self._dependencies.values()),
                'components_with_dependencies': len([t for t, deps in self._dependencies.items() if deps])
            })
            
            if stats['total_creations'] > 0:
                stats['creation_success_rate'] = 1 - (stats['creation_errors'] / stats['total_creations'])
            else:
                stats['creation_success_rate'] = 1.0
        
        return stats
    
    def export_registry(self) -> Dict[str, Any]:
        """
        Экспортирует состояние реестра
        
        Returns:
            Словарь с информацией о реестре
        """
        with self._lock:
            return {
                'types': {
                    name: {
                        'class_name': cls.__name__,
                        'module': cls.__module__,
                        'metadata': self._metadata[name].to_dict(),
                        'dependencies': self._dependencies.get(name, [])
                    }
                    for name, cls in self._components.items()
                },
                'instances': {
                    name: {
                        'type': instance.__class__.__name__,
                        'config': getattr(instance, 'config', {}),
                        'stats': instance.get_stats() if hasattr(instance, 'get_stats') else {}
                    }
                    for name, instance in self._instances.items()
                },
                'stats': self.get_stats()
            }
    
    def clear_instances(self, component_type: Optional[str] = None) -> int:
        """
        Очищает экземпляры
        
        Args:
            component_type: Тип компонентов для удаления (None для всех)
            
        Returns:
            Количество удаленных экземпляров
        """
        with self._lock:
            if component_type is None:
                # Удаляем все экземпляры
                count = len(self._instances)
                self._instances.clear()
                logger.info(f"Удалены все экземпляры: {count}")
                return count
            else:
                # Удаляем экземпляры определенного типа
                if component_type not in self._components:
                    return 0
                
                target_class = self._components[component_type]
                to_remove = [name for name, instance in self._instances.items()
                           if isinstance(instance, target_class)]
                
                for name in to_remove:
                    del self._instances[name]
                
                logger.info(f"Удалены экземпляры типа '{component_type}': {len(to_remove)}")
                return len(to_remove)
    
    def _create_metadata_from_class(self, name: str, 
                                   component_class: Type[Component]) -> ComponentInfo:
        """Создает метаданные из класса компонента"""
        # Получаем supported_modalities если есть
        supported_modalities = []
        if hasattr(component_class, 'supported_modalities'):
            supported_modalities = getattr(component_class, 'supported_modalities')
        
        # Анализируем конструктор для параметров
        try:
            sig = inspect.signature(component_class.__init__)
            parameters = {}
            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'name']:
                    continue
                parameters[param_name] = {
                    'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                }
        except Exception:
            parameters = {}
        
        return ComponentInfo(
            name=name,
            type=component_class.__name__,
            description=component_class.__doc__ or "",
            supported_modalities=supported_modalities,
            parameters=parameters
        )
    
    def _validate_dependencies(self, component_type: str) -> None:
        """Валидирует зависимости компонента"""
        dependencies = self._dependencies.get(component_type, [])
        
        for dep in dependencies:
            if dep not in self._components:
                raise ComponentError(f"Зависимость '{dep}' для компонента '{component_type}' не найдена")
    
    def _execute_creation_hooks(self, component_type: str, phase: str,
                               instance_name: str, config: Dict[str, Any]) -> None:
        """Выполняет хуки создания компонента"""
        hook_key = f"{component_type}_{phase}"
        hooks = self._creation_hooks.get(hook_key, [])
        
        for hook in hooks:
            try:
                hook(instance_name, config)
            except Exception as e:
                logger.warning(f"Ошибка в {phase}-хуке для '{component_type}': {e}")
    
    def __len__(self) -> int:
        """Возвращает количество зарегистрированных типов"""
        return len(self._components)
    
    def __contains__(self, component_type: str) -> bool:
        """Проверяет наличие типа компонента"""
        return component_type in self._components
    
    def __str__(self) -> str:
        with self._lock:
            return (f"ComponentRegistry(types={len(self._components)}, "
                   f"instances={len(self._instances)})")
    
    def __repr__(self) -> str:
        return self.__str__()


# Глобальный экземпляр реестра
GLOBAL_REGISTRY = ComponentRegistry()


def register_component(name: str, dependencies: Optional[List[str]] = None):
    """
    Декоратор для автоматической регистрации компонентов
    
    Args:
        name: Имя для регистрации
        dependencies: Список зависимостей
    
    Example:
        @register_component("MyTokenizer", dependencies=["SomeOtherComponent"])
        class MyTokenizer(Component):
            pass
    """
    def decorator(cls):
        GLOBAL_REGISTRY.register(name, cls, dependencies=dependencies)
        return cls
    return decorator


def get_component(instance_name: str) -> Optional[Component]:
    """
    Удобная функция для получения компонента из глобального реестра
    
    Args:
        instance_name: Имя экземпляра
        
    Returns:
        Экземпляр компонента или None
    """
    return GLOBAL_REGISTRY.get(instance_name)


def create_component(instance_name: str, component_type: str,
                    config: Dict[str, Any] = None) -> Component:
    """
    Удобная функция для создания компонента в глобальном реестре
    
    Args:
        instance_name: Имя экземпляра
        component_type: Тип компонента
        config: Конфигурация
        
    Returns:
        Созданный экземпляр
    """
    return GLOBAL_REGISTRY.create(instance_name, component_type, config)


def list_available_components() -> List[str]:
    """
    Возвращает список доступных типов компонентов
    
    Returns:
        Список имен типов компонентов
    """
    return GLOBAL_REGISTRY.list_types()