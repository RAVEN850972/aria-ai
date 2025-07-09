# aria/dsl/parser.py
"""YAML/JSON парсер для конфигураций ARIA"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union

from ..core import ProcessingGraph, GraphFactory, ComponentRegistry, PipelineConfig, ComponentConfig


class PipelineDSL:
    """DSL парсер для создания пайплайнов из YAML/JSON"""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.factory = GraphFactory(registry)
        self.variables = {}
    
    def parse_yaml(self, yaml_content: str) -> ProcessingGraph:
        """Парсит YAML в граф"""
        config_data = yaml.safe_load(yaml_content)
        return self._parse_config(config_data)
    
    def parse_json(self, json_content: str) -> ProcessingGraph:
        """Парсит JSON в граф"""
        config_data = json.loads(json_content)
        return self._parse_config(config_data)
    
    def parse_file(self, file_path: Union[str, Path]) -> ProcessingGraph:
        """Парсит файл конфигурации"""
        file_path = Path(file_path)
        content = file_path.read_text(encoding='utf-8')
        
        if file_path.suffix.lower() in ['.yml', '.yaml']:
            return self.parse_yaml(content)
        elif file_path.suffix.lower() == '.json':
            return self.parse_json(content)
        else:
            raise ValueError(f"Неподдерживаемый формат: {file_path.suffix}")
    
    def _parse_config(self, config_data: Dict[str, Any]) -> ProcessingGraph:
        """Основной парсер конфигурации"""
        # Загружаем переменные
        self.variables.update(config_data.get('variables', {}))
        
        # Получаем конфигурацию пайплайна
        pipeline_data = config_data.get('pipeline', config_data)
        
        # Подставляем переменные
        pipeline_data = self._substitute_variables(pipeline_data)
        
        # Создаем конфигурацию
        config = self._build_pipeline_config(pipeline_data)
        
        # Создаем граф
        return self.factory.create_from_config(config)
    
    def _substitute_variables(self, data: Any) -> Any:
        """Подставляет переменные ${var}"""
        if isinstance(data, dict):
            return {k: self._substitute_variables(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_variables(item) for item in data]
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            var_name = data[2:-1]
            return self.variables.get(var_name, data)
        return data
    
    def _build_pipeline_config(self, pipeline_data: Dict[str, Any]) -> PipelineConfig:
        """Строит конфигурацию пайплайна"""
        components = []
        for comp_data in pipeline_data.get('components', []):
            components.append(ComponentConfig(
                name=comp_data['name'],
                type=comp_data['type'],
                params=comp_data.get('params', {}),
                dependencies=comp_data.get('inputs', []),
                outputs=comp_data.get('outputs', [])
            ))
        
        connections = []
        for conn in pipeline_data.get('connections', []):
            if isinstance(conn, list) and len(conn) == 2:
                connections.append((conn[0], conn[1]))
        
        return PipelineConfig(
            name=pipeline_data.get('name', 'dsl_pipeline'),
            components=components,
            connections=connections
        )