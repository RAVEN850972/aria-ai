# aria/core/graph.py
"""
Система графов для ARIA архитектуры
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .base import Component
from .types import ProcessingResult, ComponentError

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """
    Узел графа обработки
    
    Args:
        component: Компонент для обработки
        inputs: Список имен входных портов
        outputs: Список имен выходных портов
        state: Состояние узла
        parallel: Может ли узел выполняться параллельно
        timeout: Максимальное время выполнения (секунды)
    """
    component: Component
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    parallel: bool = True
    timeout: Optional[float] = None
    
    def __post_init__(self):
        """Валидация после создания"""
        if not self.outputs:
            # Если выходы не указаны, создаем стандартный
            self.outputs = [f"{self.component.name}_output"]


class ProcessingGraph:
    """
    Граф обработки данных (DAG - Directed Acyclic Graph)
    
    Обеспечивает:
    - Топологическую сортировку для оптимального порядка выполнения
    - Параллельное выполнение независимых узлов
    - Управление потоком данных между компонентами
    - Обработку ошибок и восстановление
    """
    
    def __init__(self, name: str = "processing_graph"):
        """
        Инициализация графа
        
        Args:
            name: Имя графа
        """
        self.name = name
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, List[str]] = {}  # adjacency list
        self.reverse_edges: Dict[str, List[str]] = {}  # reverse adjacency list
        self.data_flow: Dict[str, Any] = {}  # промежуточные данные
        self._execution_order: List[str] = []
        self._execution_levels: List[List[str]] = []  # Узлы по уровням для параллельного выполнения
        self._lock = threading.Lock()
        self._stats = {
            'total_executions': 0,
            'total_time': 0.0,
            'successful_executions': 0,
            'failed_executions': 0,
            'parallel_executions': 0
        }
        
        logger.debug(f"Создан граф обработки: {name}")
    
    def add_node(self, name: str, component: Component, 
                 inputs: List[str] = None, outputs: List[str] = None,
                 parallel: bool = True, timeout: Optional[float] = None) -> None:
        """
        Добавляет узел в граф
        
        Args:
            name: Уникальное имя узла
            component: Компонент для обработки
            inputs: Список входных портов
            outputs: Список выходных портов
            parallel: Может ли выполняться параллельно
            timeout: Таймаут выполнения
        """
        if name in self.nodes:
            raise ComponentError(f"Узел с именем '{name}' уже существует")
        
        self.nodes[name] = GraphNode(
            component=component,
            inputs=inputs or [],
            outputs=outputs or [f"{name}_output"],
            parallel=parallel,
            timeout=timeout
        )
        
        # Инициализируем списки смежности
        self.edges[name] = []
        self.reverse_edges[name] = []
        
        # Сбрасываем кэшированный порядок выполнения
        self._execution_order = []
        self._execution_levels = []
        
        logger.debug(f"Добавлен узел '{name}' с компонентом {component.__class__.__name__}")
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """
        Добавляет связь между узлами
        
        Args:
            from_node: Исходный узел
            to_node: Целевой узел
        """
        if from_node not in self.nodes:
            raise ComponentError(f"Узел '{from_node}' не найден")
        if to_node not in self.nodes:
            raise ComponentError(f"Узел '{to_node}' не найден")
        
        # Проверяем на циклы перед добавлением
        if self._would_create_cycle(from_node, to_node):
            raise ComponentError(f"Добавление связи {from_node} -> {to_node} создаст цикл")
        
        if to_node not in self.edges[from_node]:
            self.edges[from_node].append(to_node)
            self.reverse_edges[to_node].append(from_node)
            
            # Сбрасываем кэшированный порядок
            self._execution_order = []
            self._execution_levels = []
            
            logger.debug(f"Добавлена связь: {from_node} -> {to_node}")
    
    def remove_node(self, name: str) -> None:
        """Удаляет узел из графа"""
        if name not in self.nodes:
            raise ComponentError(f"Узел '{name}' не найден")
        
        # Удаляем все связи с этим узлом
        for node in list(self.edges[name]):
            self.remove_edge(name, node)
        
        for node in list(self.reverse_edges[name]):
            self.remove_edge(node, name)
        
        # Удаляем узел
        del self.nodes[name]
        del self.edges[name]
        del self.reverse_edges[name]
        
        # Сбрасываем кэш
        self._execution_order = []
        self._execution_levels = []
        
        logger.debug(f"Удален узел '{name}'")
    
    def remove_edge(self, from_node: str, to_node: str) -> None:
        """Удаляет связь между узлами"""
        if from_node in self.edges and to_node in self.edges[from_node]:
            self.edges[from_node].remove(to_node)
            self.reverse_edges[to_node].remove(from_node)
            
            # Сбрасываем кэш
            self._execution_order = []
            self._execution_levels = []
            
            logger.debug(f"Удалена связь: {from_node} -> {to_node}")
    
    def _would_create_cycle(self, from_node: str, to_node: str) -> bool:
        """Проверяет, создаст ли добавление связи цикл"""
        # Используем DFS для поиска пути от to_node к from_node
        visited = set()
        
        def dfs(current: str) -> bool:
            if current == from_node:
                return True
            if current in visited:
                return False
            
            visited.add(current)
            for neighbor in self.edges.get(current, []):
                if dfs(neighbor):
                    return True
            return False
        
        return dfs(to_node)
    
    def topological_sort(self) -> List[str]:
        """
        Топологическая сортировка для определения порядка выполнения
        
        Returns:
            Список узлов в порядке выполнения
        """
        if self._execution_order:
            return self._execution_order.copy()
        
        visited = set()
        temp_visited = set()
        result = []
        
        def dfs(node: str) -> None:
            if node in temp_visited:
                raise ComponentError(f"Циклическая зависимость обнаружена в узле: {node}")
            if node in visited:
                return
            
            temp_visited.add(node)
            for neighbor in self.edges.get(node, []):
                dfs(neighbor)
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        # Обрабатываем все узлы
        for node in self.nodes:
            if node not in visited:
                dfs(node)
        
        # Обращаем для правильного порядка
        self._execution_order = result[::-1]
        return self._execution_order.copy()
    
    def get_execution_levels(self) -> List[List[str]]:
        """
        Определяет уровни выполнения для параллельной обработки
        
        Returns:
            Список уровней, каждый содержит узлы, которые можно выполнять параллельно
        """
        if self._execution_levels:
            return [level.copy() for level in self._execution_levels]
        
        # Вычисляем уровни на основе зависимостей
        levels = []
        processed = set()
        remaining = set(self.nodes.keys())
        
        while remaining:
            # Найти узлы без необработанных зависимостей
            current_level = []
            for node in list(remaining):
                dependencies = self.reverse_edges.get(node, [])
                if all(dep in processed for dep in dependencies):
                    current_level.append(node)
                    remaining.remove(node)
            
            if not current_level:
                # Если не можем найти узлы без зависимостей, есть цикл
                raise ComponentError("Обнаружен цикл в графе зависимостей")
            
            levels.append(current_level)
            processed.update(current_level)
        
        self._execution_levels = levels
        return [level.copy() for level in levels]
    
    def execute(self, input_data: Dict[str, Any], 
                parallel: bool = True, max_workers: int = 4) -> Dict[str, Any]:
        """
        Выполняет граф обработки
        
        Args:
            input_data: Входные данные
            parallel: Использовать параллельное выполнение
            max_workers: Максимальное количество потоков
            
        Returns:
            Словарь результатов выполнения
        """
        start_time = time.time()
        
        with self._lock:
            self._stats['total_executions'] += 1
            if parallel:
                self._stats['parallel_executions'] += 1
        
        # Инициализируем поток данных
        self.data_flow.clear()
        self.data_flow.update(input_data)
        results = {}
        
        try:
            if parallel and max_workers > 1:
                results = self._execute_parallel(max_workers)
            else:
                results = self._execute_sequential()
            
            # Обновляем статистику успеха
            with self._lock:
                self._stats['successful_executions'] += 1
                
        except Exception as e:
            with self._lock:
                self._stats['failed_executions'] += 1
            logger.error(f"Ошибка выполнения графа {self.name}: {e}")
            raise
        
        finally:
            execution_time = time.time() - start_time
            with self._lock:
                self._stats['total_time'] += execution_time
            
            logger.debug(f"Граф {self.name} выполнен за {execution_time:.3f}s")
        
        return results
    
    def _execute_sequential(self) -> Dict[str, Any]:
        """Последовательное выполнение графа"""
        execution_order = self.topological_sort()
        results = {}
        
        for node_name in execution_order:
            node = self.nodes[node_name]
            result = self._execute_node(node_name, node)
            results[node_name] = result
            
            # Прерываем выполнение при критической ошибке
            if result["status"] == "error" and not self._is_error_recoverable(result):
                logger.warning(f"Критическая ошибка в узле {node_name}, прерывание выполнения")
                break
        
        return results
    
    def _execute_parallel(self, max_workers: int) -> Dict[str, Any]:
        """Параллельное выполнение графа по уровням"""
        execution_levels = self.get_execution_levels()
        results = {}
        
        for level_idx, level_nodes in enumerate(execution_levels):
            logger.debug(f"Выполнение уровня {level_idx}: {level_nodes}")
            
            # Определяем узлы для параллельного выполнения
            parallel_nodes = [name for name in level_nodes 
                            if self.nodes[name].parallel]
            sequential_nodes = [name for name in level_nodes 
                              if not self.nodes[name].parallel]
            
            # Сначала выполняем последовательные узлы
            for node_name in sequential_nodes:
                node = self.nodes[node_name]
                result = self._execute_node(node_name, node)
                results[node_name] = result
            
            # Затем параллельные узлы
            if parallel_nodes:
                with ThreadPoolExecutor(max_workers=min(max_workers, len(parallel_nodes))) as executor:
                    # Запускаем задачи
                    future_to_node = {
                        executor.submit(self._execute_node, name, self.nodes[name]): name
                        for name in parallel_nodes
                    }
                    
                    # Собираем результаты
                    for future in as_completed(future_to_node):
                        node_name = future_to_node[future]
                        try:
                            result = future.result()
                            results[node_name] = result
                        except Exception as e:
                            logger.error(f"Ошибка в параллельном узле {node_name}: {e}")
                            results[node_name] = {
                                "status": "error",
                                "error": str(e),
                                "processing_time": 0.0
                            }
            
            # Проверяем критические ошибки на уровне
            critical_errors = [name for name, result in results.items()
                             if result["status"] == "error" and not self._is_error_recoverable(result)]
            
            if critical_errors:
                logger.warning(f"Критические ошибки в узлах {critical_errors}, прерывание выполнения")
                break
        
        return results
    
    def _execute_node(self, node_name: str, node: GraphNode) -> Dict[str, Any]:
        """
        Выполняет отдельный узел
        
        Args:
            node_name: Имя узла
            node: Объект узла
            
        Returns:
            Результат выполнения узла
        """
        start_time = time.time()
        
        try:
            # Собираем входные данные для узла
            node_inputs = self._collect_node_inputs(node)
            
            # Выполняем компонент с таймаутом если указан
            if node.timeout:
                result = self._execute_with_timeout(node.component, node_inputs, node.timeout)
            else:
                if len(node_inputs) == 1:
                    # Один вход
                    input_value = next(iter(node_inputs.values()))
                    result = node.component.safe_process(input_value)
                else:
                    # Множественные входы
                    result = node.component.safe_process(node_inputs)
            
            processing_time = time.time() - start_time
            
            # Обрабатываем результат
            if isinstance(result, ProcessingResult):
                # Сохраняем выходные данные
                self._store_node_outputs(node, result.data)
                
                return {
                    "status": "success" if result.is_successful() else "error",
                    "result": result,
                    "processing_time": processing_time,
                    "error": result.error if not result.is_successful() else None
                }
            else:
                # Сохраняем сырые данные
                self._store_node_outputs(node, result)
                
                return {
                    "status": "success",
                    "result": result,
                    "processing_time": processing_time
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Ошибка выполнения узла {node_name}: {str(e)}"
            logger.error(error_msg)
            
            return {
                "status": "error",
                "error": error_msg,
                "processing_time": processing_time
            }
    
    def _collect_node_inputs(self, node: GraphNode) -> Dict[str, Any]:
        """Собирает входные данные для узла"""
        node_inputs = {}
        for input_name in node.inputs:
            if input_name in self.data_flow:
                node_inputs[input_name] = self.data_flow[input_name]
            else:
                logger.warning(f"Входные данные '{input_name}' не найдены в потоке данных")
        return node_inputs
    
    def _store_node_outputs(self, node: GraphNode, result_data: Any) -> None:
        """Сохраняет выходные данные узла в поток данных"""
        if isinstance(result_data, dict) and len(node.outputs) > 1:
            # Множественные выходы
            for output_name in node.outputs:
                if output_name in result_data:
                    self.data_flow[output_name] = result_data[output_name]
        else:
            # Один выход или сырые данные
            if node.outputs:
                self.data_flow[node.outputs[0]] = result_data
    
    def _execute_with_timeout(self, component: Component, 
                            inputs: Any, timeout: float) -> Any:
        """Выполняет компонент с таймаутом"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Компонент {component.name} превысил таймаут {timeout}s")
        
        # Устанавливаем обработчик таймаута (только для Unix)
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            try:
                if isinstance(inputs, dict) and len(inputs) == 1:
                    result = component.safe_process(next(iter(inputs.values())))
                else:
                    result = component.safe_process(inputs)
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        except AttributeError:
            # Windows или другая система без SIGALRM
            # Используем ThreadPoolExecutor для эмуляции таймаута
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(component.safe_process, inputs)
                try:
                    return future.result(timeout=timeout)
                except TimeoutError:
                    raise TimeoutError(f"Компонент {component.name} превысил таймаут {timeout}s")
    
    def _is_error_recoverable(self, result: Dict[str, Any]) -> bool:
        """Определяет, является ли ошибка восстанавливаемой"""
        # Простая эвристика - можно расширить
        error_msg = result.get("error", "").lower()
        
        # Критические ошибки
        if any(keyword in error_msg for keyword in ["timeout", "memory", "system", "critical"]):
            return False
        
        # Восстанавливаемые ошибки
        return True
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Возвращает информацию о графе"""
        execution_order = self.topological_sort() if self.nodes else []
        execution_levels = self.get_execution_levels() if self.nodes else []
        
        return {
            "name": self.name,
            "nodes_count": len(self.nodes),
            "edges_count": sum(len(edges) for edges in self.edges.values()),
            "execution_order": execution_order,
            "execution_levels": execution_levels,
            "nodes": {
                name: {
                    "component_type": node.component.__class__.__name__,
                    "component_name": node.component.name,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "parallel": node.parallel,
                    "timeout": node.timeout
                }
                for name, node in self.nodes.items()
            },
            "statistics": self._stats.copy()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику выполнения графа"""
        with self._lock:
            stats = self._stats.copy()
        
        if stats['total_executions'] > 0:
            stats['avg_execution_time'] = stats['total_time'] / stats['total_executions']
            stats['success_rate'] = stats['successful_executions'] / stats['total_executions']
            stats['parallel_usage'] = stats['parallel_executions'] / stats['total_executions']
        else:
            stats['avg_execution_time'] = 0.0
            stats['success_rate'] = 0.0
            stats['parallel_usage'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику"""
        with self._lock:
            self._stats = {
                'total_executions': 0,
                'total_time': 0.0,
                'successful_executions': 0,
                'failed_executions': 0,
                'parallel_executions': 0
            }
    
    def visualize(self) -> str:
        """Возвращает текстовое представление графа"""
        lines = [f"Graph: {self.name}"]
        lines.append(f"Nodes: {len(self.nodes)}")
        lines.append(f"Edges: {sum(len(edges) for edges in self.edges.values())}")
        lines.append("")
        
        # Узлы
        lines.append("Nodes:")
        for name, node in self.nodes.items():
            comp_type = node.component.__class__.__name__
            lines.append(f"  {name}: {comp_type}")
            if node.inputs:
                lines.append(f"    inputs: {node.inputs}")
            if node.outputs:
                lines.append(f"    outputs: {node.outputs}")
        
        lines.append("")
        
        # Связи
        lines.append("Connections:")
        for from_node, to_nodes in self.edges.items():
            for to_node in to_nodes:
                lines.append(f"  {from_node} -> {to_node}")
        
        if self._execution_order:
            lines.append("")
            lines.append(f"Execution order: {' -> '.join(self._execution_order)}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return f"ProcessingGraph(name='{self.name}', nodes={len(self.nodes)}, edges={sum(len(e) for e in self.edges.values())})"
    
    def __repr__(self) -> str:
        return self.__str__()