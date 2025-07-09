# aria/dsl/templates.py
"""Готовые шаблоны пайплайнов ARIA"""

from typing import Dict, Any

from ..core import ProcessingGraph, ComponentRegistry, GraphFactory, PipelineConfig, ComponentConfig


# Встроенные шаблоны
BUILTIN_TEMPLATES = {
    'simple_text_generation': {
        'name': 'simple_text_generation',
        'components': [
            {
                'name': 'tokenizer',
                'type': 'AdvancedTextTokenizer',
                'params': {'preserve_punctuation': True},
                'inputs': ['text_input'],
                'outputs': ['tokens']
            },
            {
                'name': 'ngram_extractor', 
                'type': 'NGramExtractor',
                'params': {'ngram': {'min_n': 2, 'max_n': 4}},
                'inputs': ['tokens'],
                'outputs': ['ngram_data']
            },
            {
                'name': 'generator',
                'type': 'BeamSearchGenerator', 
                'params': {'generation': {'beam_size': 3, 'max_length': 15}},
                'inputs': ['tokens'],
                'outputs': ['generated_text']
            }
        ],
        'connections': [
            ['tokenizer', 'ngram_extractor'],
            ['tokenizer', 'generator']
        ]
    },
    
    'advanced_generation': {
        'name': 'advanced_generation',
        'components': [
            {
                'name': 'tokenizer',
                'type': 'AdvancedTextTokenizer',
                'params': {'preserve_punctuation': True, 'min_token_length': 1},
                'inputs': ['text_input'],
                'outputs': ['tokens']
            },
            {
                'name': 'ngram_extractor',
                'type': 'NGramExtractor', 
                'params': {'ngram': {'min_n': 2, 'max_n': 5, 'dynamic_selection': True}},
                'inputs': ['tokens'],
                'outputs': ['ngram_data']
            },
            {
                'name': 'generator',
                'type': 'BeamSearchGenerator',
                'params': {'generation': {'beam_size': 5, 'max_length': 25, 'temperature': 0.8}},
                'inputs': ['tokens'],
                'outputs': ['generated_text']
            }
        ],
        'connections': [
            ['tokenizer', 'ngram_extractor'],
            ['tokenizer', 'generator']
        ]
    }
}


class TemplateManager:
    """Управляет шаблонами пайплайнов"""
    
    def __init__(self):
        self.templates = BUILTIN_TEMPLATES.copy()
    
    def get_template(self, name: str) -> Dict[str, Any]:
        """Получает шаблон по имени"""
        if name not in self.templates:
            raise ValueError(f"Шаблон '{name}' не найден")
        return self.templates[name].copy()
    
    def list_templates(self) -> list:
        """Список доступных шаблонов"""
        return list(self.templates.keys())
    
    def create_from_template(self, template_name: str, registry: ComponentRegistry,
                           overrides: Dict[str, Any] = None) -> ProcessingGraph:
        """Создает граф из шаблона"""
        template = self.get_template(template_name)
        
        # Применяем переопределения
        if overrides:
            template = self._apply_overrides(template, overrides)
        
        # Создаем конфигурацию
        components = [ComponentConfig(**comp) for comp in template['components']]
        connections = template.get('connections', [])
        
        config = PipelineConfig(
            name=template_name,
            components=components,
            connections=connections
        )
        
        # Создаем граф
        factory = GraphFactory(registry)
        graph = factory.create_from_config(config)
        
        # Специальная настройка для генераторов
        self._setup_generators(graph, registry)
        
        return graph
    
    def _apply_overrides(self, template: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Применяет переопределения к шаблону"""
        import copy
        result = copy.deepcopy(template)
        
        for path, value in overrides.items():
            keys = path.split('.')
            current = result
            
            # Навигируем к нужному месту
            for key in keys[:-1]:
                if key.isdigit():
                    current = current[int(key)]
                else:
                    current = current[key]
            
            # Устанавливаем значение
            final_key = keys[-1]
            if final_key.isdigit():
                current[int(final_key)] = value
            else:
                current[final_key] = value
        
        return result
    
    def _setup_generators(self, graph: ProcessingGraph, registry: ComponentRegistry):
        """Настраивает связи между генераторами и n-gram экстракторами"""
        # Находим компоненты
        ngram_extractors = {}
        generators = {}
        
        for node_name, node in graph.nodes.items():
            comp_type = node.component.__class__.__name__
            if comp_type == "NGramExtractor":
                ngram_extractors[node_name] = node.component
            elif comp_type == "BeamSearchGenerator":
                generators[node_name] = node.component
        
        # Связываем генераторы с экстракторами
        for gen_name, generator in generators.items():
            for ext_name, extractor in ngram_extractors.items():
                generator.set_ngram_model(extractor)
                break  # Берем первый доступный


def get_builtin_templates() -> Dict[str, Any]:
    """Возвращает встроенные шаблоны"""
    return BUILTIN_TEMPLATES.copy()


def create_text_pipeline(registry: ComponentRegistry, 
                        template_name: str = "simple_text_generation",
                        **overrides) -> ProcessingGraph:
    """Быстрое создание текстового пайплайна"""
    manager = TemplateManager()
    return manager.create_from_template(template_name, registry, overrides)