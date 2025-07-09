# aria/dsl/fluent.py
"""Fluent API для создания пайплайнов ARIA"""

from typing import Dict, Any

from ..core import ProcessingGraph, ComponentRegistry


class FluentPipelineBuilder:
    """Fluent API для программного создания пайплайнов"""
    
    def __init__(self, registry: ComponentRegistry, name: str = "fluent_pipeline"):
        self.registry = registry
        self.graph = ProcessingGraph(name)
        self.last_component = None
    
    def add_component(self, component_type: str, name: str, **params) -> 'FluentPipelineBuilder':
        """Добавляет компонент"""
        component = self.registry.create(name, component_type, params)
        
        # Правильные входы и выходы
        if self.last_component is None:
            # Первый компонент
            inputs = ["input"]  # Стандартный вход
        else:
            inputs = [f"{self.last_component}_output"]
        
        outputs = [f"{name}_output"]
        
        self.graph.add_node(name, component, inputs, outputs)
        
        if self.last_component:
            self.graph.add_edge(self.last_component, name)
        
        self.last_component = name
        return self
    
    def tokenizer(self, name: str = "tokenizer", **params) -> 'FluentPipelineBuilder':
        """Добавляет токенизатор"""
        return self.add_component("AdvancedTextTokenizer", name, **params)
    
    def ngram_extractor(self, name: str = "ngram_extractor", **params) -> 'FluentPipelineBuilder':
        """Добавляет извлекатель n-грамм"""
        extractor = self.add_component("NGramExtractor", name, **params)
        
        # Связываем с генератором если есть
        if hasattr(self, '_pending_generator'):
            generator_name, generator_params = self._pending_generator
            generator = self.registry.create(generator_name, "BeamSearchGenerator", generator_params)
            generator.set_ngram_model(self.registry.get(name))
            delattr(self, '_pending_generator')
        
        return extractor
    
    def generator(self, name: str = "generator", **params) -> 'FluentPipelineBuilder':
        """Добавляет генератор"""
        # Если n-gram extractor уже есть, связываем сразу
        ngram_name = None
        for node_name, node in self.graph.nodes.items():
            if node.component.__class__.__name__ == "NGramExtractor":
                ngram_name = node_name
                break
        
        generator = self.add_component("BeamSearchGenerator", name, **params)
        
        if ngram_name:
            generator_component = self.registry.get(name)
            ngram_component = self.registry.get(ngram_name)
            generator_component.set_ngram_model(ngram_component)
        else:
            # Сохраняем для связывания позже
            self._pending_generator = (name, params)
        
        return generator
    
    def fusion(self, name: str = "fusion", method: str = "concatenation", **params) -> 'FluentPipelineBuilder':
        """Добавляет компонент слияния"""
        params['method'] = method
        return self.add_component("TextFusionStrategy", name, **params)
    
    def connect(self, from_name: str, to_name: str) -> 'FluentPipelineBuilder':
        """Явно соединяет компоненты"""
        self.graph.add_edge(from_name, to_name)
        return self
    
    def build(self) -> ProcessingGraph:
        """Строит финальный граф"""
        return self.graph