# aria/core/factory.py
"""
Фабрика для создания графов обработки из конфигураций
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

from .graph import ProcessingGraph
from .registry import ComponentRegistry
from .types import ComponentError, ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ComponentConfig:
    """
    Конфигурация отдельного компонента
    
    Args:
        name: Уникальное имя экземпляра компонента
        type: Тип компонента (должен быть зарегистрирован)
        params: Параметры конфигурации компонента
        dependencies: Список входных портов (зависимостей)
        outputs: Список выходных портов
        parallel: Может ли выполняться параллельно
        timeout: Таймаут выполнения в секундах
        enabled: Включен ли компонент
        description: Описание компонента
    """
    name: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    parallel: bool = True
    timeout: Optional[float] = None
    enabled: bool = True
    description: str = ""
    
    def __post_init__(self):
        """Валидация после создания"""
        if not self.name:
            raise ConfigurationError("Имя компонента не может быть пустым")
        if not self.type:
            raise ConfigurationError("Тип компонента не может быть пустым")
        
        # Устанавливаем выходы по умолчанию если не указаны
        if not self.outputs:
            self.outputs = [f"{self.name}_output"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь"""
        return {
            'name': self.name,
            'type': self.type,
            'params': self.params,
            'dependencies': self.dependencies,
            'outputs': self.outputs,
            'parallel': self.parallel,
            'timeout': self.timeout,
            'enabled': self.enabled,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentConfig':
        """Создает из словаря"""
        return cls(
            name=data['name'],
            type=data['type'],
            params=data.get('params', {}),
            dependencies=data.get('dependencies', []),
            outputs=data.get('outputs', []),
            parallel=data.get('parallel', True),
            timeout=data.get('timeout'),
            enabled=data.get('enabled', True),
            description=data.get('description', '')
        )


@dataclass
class PipelineConfig:
    """
    Конфигурация пайплайна обработки
    
    Args:
        name: Имя пайплайна
        components: Список конфигураций компонентов
        connections: Список связей между компонентами
        global_config: Глобальная конфигурация пайплайна
        parallel_execution: Использовать параллельное выполнение
        max_workers: Максимальное количество потоков
        error_handling: Стратегия обработки ошибок
        description: Описание пайплайна
        version: Версия конфигурации
    """
    name: str
    components: List[ComponentConfig]
    connections: List[Tuple[str, str]] = field(default_factory=list)
    global_config: Dict[str, Any] = field(default_factory=dict)
    parallel_execution: bool = True
    max_workers: int = 4
    error_handling: str = "continue"  # "continue", "stop", "retry"
    description: str = ""
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Валидация после создания"""
        if not self.name:
            raise ConfigurationError("Имя пайплайна не может быть пустым")
        if not self.components:
            raise ConfigurationError("Пайплайн должен содержать хотя бы один компонент")
        
        # Проверяем уникальность имен компонентов
        component_names = [comp.name for comp in self.components]
        if len(component_names) != len(set(component_names)):
            duplicates = [name for name in component_names if component_names.count(name) > 1]
            raise ConfigurationError(f"Дублирующиеся имена компонентов: {duplicates}")
        
        # Валидируем связи
        self._validate_connections()
        
        # Валидируем параметры
        if self.max_workers < 1:
            raise ConfigurationError("max_workers должен быть больше 0")
        
        if self.error_handling not in ["continue", "stop", "retry"]:
            raise ConfigurationError("error_handling должен быть 'continue', 'stop' или 'retry'")
    
    def _validate_connections(self):
        """Валидирует связи между компонентами"""
        component_names = {comp.name for comp in self.components}
        
        for from_comp, to_comp in self.connections:
            if from_comp not in component_names:
                raise ConfigurationError(f"Компонент '{from_comp}' в связи не найден")
            if to_comp not in component_names:
                raise ConfigurationError(f"Компонент '{to_comp}' в связи не найден")
    
    def get_component_by_name(self, name: str) -> Optional[ComponentConfig]:
        """Получает конфигурацию компонента по имени"""
        for comp in self.components:
            if comp.name == name:
                return comp
        return None
    
    def get_enabled_components(self) -> List[ComponentConfig]:
        """Возвращает только включенные компоненты"""
        return [comp for comp in self.components if comp.enabled]
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь"""
        return {
            'name': self.name,
            'components': [comp.to_dict() for comp in self.components],
            'connections': [[from_comp, to_comp] for from_comp, to_comp in self.connections],
            'global_config': self.global_config,
            'parallel_execution': self.parallel_execution,
            'max_workers': self.max_workers,
            'error_handling': self.error_handling,
            'description': self.description,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Создает из словаря"""
        components = [ComponentConfig.from_dict(comp_data) 
                     for comp_data in data.get('components', [])]
        
        connections = []
        for conn_data in data.get('connections', []):
            if isinstance(conn_data, list) and len(conn_data) == 2:
                connections.append((conn_data[0], conn_data[1]))
            elif isinstance(conn_data, dict):
                connections.append((conn_data['from'], conn_data['to']))
        
        return cls(
            name=data['name'],
            components=components,
            connections=connections,
            global_config=data.get('global_config', {}),
            parallel_execution=data.get('parallel_execution', True),
            max_workers=data.get('max_workers', 4),
            error_handling=data.get('error_handling', 'continue'),
            description=data.get('description', ''),
            version=data.get('version', '1.0.0')
        )


class GraphFactory:
    """
    Фабрика для создания графов обработки из конфигураций
    
    Обеспечивает:
    - Валидацию конфигураций
    - Создание компонентов
    - Построение графа зависимостей
    - Оптимизацию выполнения
    """
    
    def __init__(self, registry: ComponentRegistry):
        """
        Инициализация фабрики
        
        Args:
            registry: Реестр компонентов
        """
        self.registry = registry
        self._validation_hooks = []
        self._creation_hooks = []
        
        logger.debug("Инициализирована фабрика графов")
    
    def create_from_config(self, config: PipelineConfig) -> ProcessingGraph:
        """
        Создает граф обработки из конфигурации
        
        Args:
            config: Конфигурация пайплайна
            
        Returns:
            Созданный граф обработки
        """
        logger.info(f"Создание графа '{config.name}' из конфигурации")
        
        # Валидация конфигурации
        self._validate_config(config)
        
        # Создаем граф
        graph = ProcessingGraph(config.name)
        
        try:
            # Создаем компоненты
            self._create_components(graph, config)
            
            # Создаем связи
            self._create_connections(graph, config)
            
            # Применяем глобальную конфигурацию
            self._apply_global_config(graph, config)
            
            # Выполняем хуки создания
            self._execute_creation_hooks(graph, config)
            
            # Валидируем созданный граф
            self._validate_created_graph(graph)
            
            logger.info(f"Граф '{config.name}' успешно создан: "
                       f"{len(graph.nodes)} узлов, "
                       f"{sum(len(edges) for edges in graph.edges.values())} связей")
            
            return graph
            
        except Exception as e:
            error_msg = f"Ошибка создания графа '{config.name}': {e}"
            logger.error(error_msg)
            raise ComponentError(error_msg) from e
    
    def create_simple_pipeline(self, *component_specs: Tuple[str, str, Dict[str, Any]]) -> ProcessingGraph:
        """
        Создает простой последовательный пайплайн
        
        Args:
            component_specs: Кортежи (name, type, config) для каждого компонента
            
        Returns:
            Созданный граф
            
        Example:
            factory.create_simple_pipeline(
                ("tokenizer", "TextTokenizer", {"lang": "ru"}),
                ("processor", "TextProcessor", {}),
                ("generator", "TextGenerator", {"length": 50})
            )
        """
        if not component_specs:
            raise ConfigurationError("Должен быть указан хотя бы один компонент")
        
        # Создаем конфигурации компонентов
        components = []
        connections = []
        
        prev_name = None
        for i, (name, comp_type, comp_config) in enumerate(component_specs):
            inputs = [f"{prev_name}_output"] if prev_name else ["input"]
            outputs = [f"{name}_output"]
            
            components.append(ComponentConfig(
                name=name,
                type=comp_type,
                params=comp_config,
                dependencies=inputs,
                outputs=outputs
            ))
            
            if prev_name:
                connections.append((prev_name, name))
            
            prev_name = name
        
        # Создаем конфигурацию пайплайна
        pipeline_config = PipelineConfig(
            name="simple_pipeline",
            components=components,
            connections=connections
        )
        
        return self.create_from_config(pipeline_config)
    
    def add_validation_hook(self, hook: callable) -> None:
        """
        Добавляет хук валидации конфигурации
        
        Args:
            hook: Функция валидации, принимающая PipelineConfig
        """
        self._validation_hooks.append(hook)
        logger.debug("Добавлен хук валидации")
    
    def add_creation_hook(self, hook: callable) -> None:
        """
        Добавляет хук создания графа
        
        Args:
            hook: Функция, принимающая (ProcessingGraph, PipelineConfig)
        """
        self._creation_hooks.append(hook)
        logger.debug("Добавлен хук создания")
    
    def _validate_config(self, config: PipelineConfig) -> None:
        """Валидирует конфигурацию пайплайна"""
        logger.debug(f"Валидация конфигурации '{config.name}'")
        
        # Проверяем доступность типов компонентов
        for comp_config in config.components:
            if comp_config.type not in self.registry:
                available_types = self.registry.list_types()
                raise ConfigurationError(
                    f"Тип компонента '{comp_config.type}' не зарегистрирован. "
                    f"Доступные типы: {available_types}"
                )
        
        # Проверяем зависимости
        violations = self.registry.validate_all_dependencies()
        if violations:
            raise ConfigurationError(f"Нарушения зависимостей: {violations}")
        
        # Выполняем пользовательские хуки валидации
        for hook in self._validation_hooks:
            try:
                hook(config)
            except Exception as e:
                raise ConfigurationError(f"Ошибка в хуке валидации: {e}") from e
    
    def _create_components(self, graph: ProcessingGraph, config: PipelineConfig) -> None:
        """Создает компоненты и добавляет их в граф"""
        logger.debug(f"Создание {len(config.components)} компонентов")
        
        for comp_config in config.get_enabled_components():
            # Объединяем локальную и глобальную конфигурацию
            merged_config = config.global_config.copy()
            merged_config.update(comp_config.params)
            
            # Создаем компонент
            component = self.registry.create(
                comp_config.name,
                comp_config.type,
                merged_config
            )
            
            # Добавляем в граф
            graph.add_node(
                comp_config.name,
                component,
                comp_config.dependencies,
                comp_config.outputs,
                comp_config.parallel,
                comp_config.timeout
            )
            
            logger.debug(f"Создан компонент '{comp_config.name}' типа '{comp_config.type}'")
    
    def _create_connections(self, graph: ProcessingGraph, config: PipelineConfig) -> None:
        """Создает связи между компонентами"""
        logger.debug(f"Создание {len(config.connections)} связей")
        
        for from_comp, to_comp in config.connections:
            # Проверяем что оба компонента включены
            from_config = config.get_component_by_name(from_comp)
            to_config = config.get_component_by_name(to_comp)
            
            if from_config and from_config.enabled and to_config and to_config.enabled:
                graph.add_edge(from_comp, to_comp)
                logger.debug(f"Создана связь: {from_comp} -> {to_comp}")
    
    def _apply_global_config(self, graph: ProcessingGraph, config: PipelineConfig) -> None:
        """Применяет глобальную конфигурацию к графу"""
        # Сохраняем метаданные конфигурации в графе
        graph.data_flow['_pipeline_config'] = config.global_config
        graph.data_flow['_error_handling'] = config.error_handling
        graph.data_flow['_max_workers'] = config.max_workers
        graph.data_flow['_parallel_execution'] = config.parallel_execution
        
        logger.debug("Применена глобальная конфигурация")
    
    def _execute_creation_hooks(self, graph: ProcessingGraph, config: PipelineConfig) -> None:
        """Выполняет хуки создания графа"""
        for hook in self._creation_hooks:
            try:
                hook(graph, config)
            except Exception as e:
                logger.warning(f"Ошибка в хуке создания: {e}")
    
    def _validate_created_graph(self, graph: ProcessingGraph) -> None:
        """Валидирует созданный граф"""
        # Проверяем что граф не пустой
        if not graph.nodes:
            raise ComponentError("Созданный граф не содержит узлов")
        
        # Проверяем связность (что есть хотя бы один путь)
        try:
            execution_order = graph.topological_sort()
            if not execution_order:
                raise ComponentError("Граф не имеет допустимого порядка выполнения")
        except Exception as e:
            raise ComponentError(f"Граф содержит циклы или другие ошибки структуры: {e}")
        
        logger.debug("Созданный граф прошел валидацию")


def create_graph_from_dict(graph_data: Dict[str, Any], 
                          registry: ComponentRegistry) -> ProcessingGraph:
    """
    Удобная функция для создания графа из словаря
    
    Args:
        graph_data: Данные конфигурации
        registry: Реестр компонентов
        
    Returns:
        Созданный граф
    """
    config = PipelineConfig.from_dict(graph_data)
    factory = GraphFactory(registry)
    return factory.create_from_config(config)


def create_simple_text_pipeline(registry: ComponentRegistry,
                               tokenizer_config: Dict[str, Any] = None,
                               processor_config: Dict[str, Any] = None,
                               generator_config: Dict[str, Any] = None) -> ProcessingGraph:
    """
    Создает простой пайплайн обработки текста
    
    Args:
        registry: Реестр компонентов
        tokenizer_config: Конфигурация токенизатора
        processor_config: Конфигурация процессора
        generator_config: Конфигурация генератора
        
    Returns:
        Созданный граф
    """
    factory = GraphFactory(registry)
    
    return factory.create_simple_pipeline(
        ("tokenizer", "AdvancedTextTokenizer", tokenizer_config or {}),
        ("processor", "NGramExtractor", processor_config or {}),
        ("generator", "BeamSearchGenerator", generator_config or {})
    )