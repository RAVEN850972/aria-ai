# aria/dsl/cli.py
"""CLI интерфейс для управления пайплайнами ARIA"""

from typing import List, Optional

from ..core import ProcessingGraph, ComponentRegistry
from .parser import PipelineDSL
from .templates import TemplateManager


class PipelineCLI:
    """CLI интерфейс для работы с пайплайнами"""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.dsl = PipelineDSL(registry)
        self.template_manager = TemplateManager()
        self.active_graph: Optional[ProcessingGraph] = None
    
    def execute_command(self, command: str, args: List[str]) -> str:
        """Выполняет CLI команду"""
        try:
            if command == "create":
                return self._create_pipeline(args)
            elif command == "load":
                return self._load_pipeline(args)
            elif command == "run":
                return self._run_pipeline(args)
            elif command == "info":
                return self._show_info()
            elif command == "templates":
                return self._list_templates()
            elif command == "components":
                return self._list_components()
            else:
                return f"Неизвестная команда: {command}"
        except Exception as e:
            return f"Ошибка: {e}"
    
    def _create_pipeline(self, args: List[str]) -> str:
        """Создает пайплайн из шаблона"""
        if not args:
            return "Укажите имя шаблона"
        
        template_name = args[0]
        
        try:
            self.active_graph = self.template_manager.create_from_template(
                template_name, self.registry
            )
            return f"✅ Пайплайн создан из шаблона '{template_name}'"
        except Exception as e:
            return f"❌ Ошибка создания: {e}"
    
    def _load_pipeline(self, args: List[str]) -> str:
        """Загружает пайплайн из файла"""
        if not args:
            return "Укажите путь к файлу"
        
        try:
            self.active_graph = self.dsl.parse_file(args[0])
            return f"✅ Пайплайн загружен из '{args[0]}'"
        except Exception as e:
            return f"❌ Ошибка загрузки: {e}"
    
    def _run_pipeline(self, args: List[str]) -> str:
        """Запускает активный пайплайн"""
        if not self.active_graph:
            return "❌ Нет активного пайплайна"
        
        if not args:
            return "❌ Укажите входной текст"
        
        text = " ".join(args)
        
        try:
            results = self.active_graph.execute({"text_input": text})
            
            # Форматируем результат
            output = ["🔄 Результаты выполнения:"]
            for node_name, result in results.items():
                status = "✅" if result["status"] == "success" else "❌"
                time_info = f"({result.get('processing_time', 0):.3f}s)"
                output.append(f"  {status} {node_name}: {time_info}")
            
            return "\n".join(output)
        except Exception as e:
            return f"❌ Ошибка выполнения: {e}"
    
    def _show_info(self) -> str:
        """Показывает информацию о пайплайне"""
        if not self.active_graph:
            return "❌ Нет активного пайплайна"
        
        info = self.active_graph.get_graph_info()
        
        output = [
            f"📊 Пайплайн: {info['name']}",
            f"   Узлов: {info['nodes_count']}",
            f"   Связей: {info['edges_count']}",
            f"   Порядок: {' -> '.join(info['execution_order'])}",
            "",
            "🧩 Компоненты:"
        ]
        
        for node_name, node_info in info['nodes'].items():
            output.append(f"  • {node_name}: {node_info['component_type']}")
        
        return "\n".join(output)
    
    def _list_templates(self) -> str:
        """Список шаблонов"""
        templates = self.template_manager.list_templates()
        output = ["📝 Доступные шаблоны:"]
        for template in templates:
            output.append(f"  • {template}")
        return "\n".join(output)
    
    def _list_components(self) -> str:
        """Список компонентов"""
        types = self.registry.list_types()
        instances = self.registry.list_instances()
        
        output = [
            "🧩 Типы компонентов:",
            *[f"  • {t}" for t in types],
            "",
            "⚙️ Активные экземпляры:",
            *[f"  • {i}" for i in instances]
        ]
        
        return "\n".join(output)