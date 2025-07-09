# aria/processors/text/fusion.py
"""
Слияние и управление моделями для текстовых данных ARIA
"""

import pickle
import os
import time
import threading
from typing import List, Dict, Any, Optional, Union
import logging

from ...core import Component, StatefulComponent, ModalityType, ProcessingResult, ProcessingStage

logger = logging.getLogger(__name__)


class TextFusionStrategy(Component):
    """
    Стратегия слияния для текстовых данных
    
    Поддерживает различные методы объединения результатов обработки текста:
    - Конкатенация (простое объединение)
    - Взвешенное усреднение по качеству
    - Голосование по частоте
    - Интеллектуальное слияние с приоритетами
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Инициализация стратегии слияния
        
        Args:
            name: Имя компонента
            config: Конфигурация слияния
        """
        super().__init__(name, config)
        
        self.fusion_method = config.get('method', 'concatenation')
        self.weight_strategy = config.get('weight_strategy', 'equal')  # 'equal', 'quality', 'frequency'
        self.quality_threshold = config.get('quality_threshold', 0.5)
        self.max_results = config.get('max_results', None)
        self.preserve_order = config.get('preserve_order', True)
        self.remove_duplicates = config.get('remove_duplicates', True)
        
        # Статистика слияния
        self.fusion_stats = {
            'total_fusions': 0,
            'avg_input_count': 0.0,
            'avg_output_size': 0.0,
            'method_usage': {},
            'duplicates_removed': 0
        }
        
        # Проверяем поддерживаемые методы
        self.supported_methods = {
            'concatenation': self._concatenate_results,
            'weighted_average': self._weighted_average_results,
            'voting': self._voting_results,
            'intelligent': self._intelligent_fusion,
            'best_quality': self._best_quality_fusion
        }
        
        if self.fusion_method not in self.supported_methods:
            raise ValueError(f"Неподдерживаемый метод слияния: {self.fusion_method}")
        
        logger.debug(f"Инициализирован TextFusionStrategy {name} с методом {self.fusion_method}")
    
    def process(self, input_data: List[ProcessingResult]) -> ProcessingResult:
        """
        Объединяет результаты обработки текста
        
        Args:
            input_data: Список ProcessingResult для объединения
            
        Returns:
            ProcessingResult с объединенными данными
        """
        if not isinstance(input_data, list):
            raise ValueError("Входные данные должны быть списком ProcessingResult")
        
        if not input_data:
            return ProcessingResult(
                data=[],
                stage=ProcessingStage.FUSION,
                modality=ModalityType.TEXT,
                metadata={"fusion_method": self.fusion_method, "input_count": 0},
                component_name=self.name
            )
        
        # Фильтруем успешные результаты
        valid_results = [result for result in input_data if result.is_successful()]
        
        if not valid_results:
            logger.warning("Нет успешных результатов для слияния")
            return ProcessingResult(
                data=[],
                stage=ProcessingStage.FUSION,
                modality=ModalityType.TEXT,
                error="Нет успешных входных результатов",
                metadata={"fusion_method": self.fusion_method, "input_count": len(input_data)},
                component_name=self.name
            )
        
        # Применяем стратегию слияния
        fusion_method = self.supported_methods[self.fusion_method]
        fused_result = fusion_method(valid_results)
        
        # Обновляем статистику
        self._update_fusion_stats(input_data, fused_result)
        
        return fused_result
    
    def _concatenate_results(self, results: List[ProcessingResult]) -> ProcessingResult:
        """
        Конкатенация результатов
        
        Args:
            results: Список результатов
            
        Returns:
            Объединенный результат
        """
        combined_data = []
        total_score = 0
        all_metadata = {}
        
        for i, result in enumerate(results):
            if isinstance(result.data, list):
                if self.preserve_order:
                    combined_data.extend(result.data)
                else:
                    # Перемешиваем для разнообразия
                    import random
                    shuffled = result.data.copy()
                    random.shuffle(shuffled)
                    combined_data.extend(shuffled)
            else:
                combined_data.append(result.data)
            
            total_score += result.score
            all_metadata[f'source_{i}'] = {
                'component': result.component_name,
                'score': result.score,
                'size': len(result.data) if hasattr(result.data, '__len__') else 1
            }
        
        # Удаляем дубликаты если требуется
        if self.remove_duplicates:
            original_size = len(combined_data)
            combined_data = list(dict.fromkeys(combined_data))  # Сохраняем порядок
            removed = original_size - len(combined_data)
            self.fusion_stats['duplicates_removed'] += removed
            all_metadata['duplicates_removed'] = removed
        
        # Ограничиваем размер если требуется
        if self.max_results and len(combined_data) > self.max_results:
            combined_data = combined_data[:self.max_results]
            all_metadata['truncated_to'] = self.max_results
        
        return ProcessingResult(
            data=combined_data,
            stage=ProcessingStage.FUSION,
            modality=ModalityType.TEXT,
            score=total_score / len(results) if results else 0,
            metadata={
                "fusion_method": "concatenation",
                "source_count": len(results),
                "sources": all_metadata,
                "final_size": len(combined_data)
            },
            component_name=self.name
        )
    
    def _weighted_average_results(self, results: List[ProcessingResult]) -> ProcessingResult:
        """
        Взвешенное усреднение результатов
        
        Args:
            results: Список результатов
            
        Returns:
            Лучший результат с учетом весов
        """
        if not results:
            return self._empty_result()
        
        # Вычисляем веса
        weights = self._calculate_weights(results)
        
        # Для текста "усреднение" означает выбор лучшего с учетом весов
        weighted_scores = []
        for i, result in enumerate(results):
            weighted_score = result.score * weights[i]
            weighted_scores.append((weighted_score, i, result))
        
        # Выбираем лучший результат
        best_score, best_idx, best_result = max(weighted_scores, key=lambda x: x[0])
        
        # Создаем метаданные о выборе
        selection_metadata = {
            "fusion_method": "weighted_average",
            "selected_index": best_idx,
            "weighted_score": best_score,
            "original_score": best_result.score,
            "weight_applied": weights[best_idx],
            "candidates": [
                {
                    "index": i,
                    "score": result.score,
                    "weight": weights[i],
                    "weighted_score": weighted_scores[i][0],
                    "component": result.component_name
                }
                for i, result in enumerate(results)
            ]
        }
        
        return ProcessingResult(
            data=best_result.data,
            stage=ProcessingStage.FUSION,
            modality=ModalityType.TEXT,
            score=best_score,
            metadata=selection_metadata,
            component_name=self.name
        )
    
    def _voting_results(self, results: List[ProcessingResult]) -> ProcessingResult:
        """
        Слияние на основе голосования
        
        Args:
            results: Список результатов
            
        Returns:
            Результат на основе голосования
        """
        from collections import Counter
        
        # Собираем все элементы для голосования
        all_items = []
        source_info = {}
        
        for i, result in enumerate(results):
            if isinstance(result.data, list):
                for item in result.data:
                    all_items.append(item)
                    if item not in source_info:
                        source_info[item] = []
                    source_info[item].append({
                        'source_index': i,
                        'component': result.component_name,
                        'score': result.score
                    })
            else:
                all_items.append(result.data)
                if result.data not in source_info:
                    source_info[result.data] = []
                source_info[result.data].append({
                    'source_index': i,
                    'component': result.component_name,
                    'score': result.score
                })
        
        # Подсчитываем голоса
        vote_counts = Counter(all_items)
        
        # Сортируем по количеству голосов и качеству
        voted_items = []
        for item, count in vote_counts.items():
            # Средний score от всех источников этого элемента
            avg_score = sum(info['score'] for info in source_info[item]) / len(source_info[item])
            combined_score = count * avg_score  # Голоса * качество
            
            voted_items.append((item, count, avg_score, combined_score))
        
        # Сортируем по комбинированному счету
        voted_items.sort(key=lambda x: x[3], reverse=True)
        
        # Берем результаты
        final_items = [item[0] for item in voted_items]
        
        if self.max_results:
            final_items = final_items[:self.max_results]
        
        # Метаданные голосования
        voting_metadata = {
            "fusion_method": "voting",
            "total_votes": len(all_items),
            "unique_items": len(vote_counts),
            "voting_results": [
                {
                    "item": str(item)[:100],  # Ограничиваем длину для читаемости
                    "votes": count,
                    "avg_score": avg_score,
                    "combined_score": combined_score,
                    "sources": len(source_info[item])
                }
                for item, count, avg_score, combined_score in voted_items[:10]  # Топ 10
            ]
        }
        
        return ProcessingResult(
            data=final_items,
            stage=ProcessingStage.FUSION,
            modality=ModalityType.TEXT,
            score=sum(item[3] for item in voted_items[:len(final_items)]) / len(final_items) if final_items else 0,
            metadata=voting_metadata,
            component_name=self.name
        )
    
    def _intelligent_fusion(self, results: List[ProcessingResult]) -> ProcessingResult:
        """
        Интеллектуальное слияние с адаптивными стратегиями
        
        Args:
            results: Список результатов
            
        Returns:
            Интеллектуально объединенный результат
        """
        # Анализируем характеристики входных данных
        analysis = self._analyze_results(results)
        
        # Выбираем оптимальную стратегию на основе анализа
        if analysis['quality_variance'] > 0.3:
            # Большая разница в качестве - выбираем лучший
            return self._best_quality_fusion(results)
        elif analysis['size_variance'] > 0.5:
            # Большая разница в размере - используем голосование
            return self._voting_results(results)
        else:
            # Результаты похожи - конкатенируем
            return self._concatenate_results(results)
    
    def _best_quality_fusion(self, results: List[ProcessingResult]) -> ProcessingResult:
        """
        Выбор результата с лучшим качеством
        
        Args:
            results: Список результатов
            
        Returns:
            Лучший результат
        """
        # Фильтруем по порогу качества
        quality_results = [r for r in results if r.score >= self.quality_threshold]
        
        if not quality_results:
            quality_results = results  # Используем все если никто не прошел порог
        
        # Выбираем лучший
        best_result = max(quality_results, key=lambda x: x.score)
        
        return ProcessingResult(
            data=best_result.data,
            stage=ProcessingStage.FUSION,
            modality=ModalityType.TEXT,
            score=best_result.score,
            metadata={
                "fusion_method": "best_quality",
                "selected_score": best_result.score,
                "selected_component": best_result.component_name,
                "candidates_count": len(results),
                "quality_threshold": self.quality_threshold,
                "passed_threshold": len(quality_results)
            },
            component_name=self.name
        )
    
    def _calculate_weights(self, results: List[ProcessingResult]) -> List[float]:
        """Вычисляет веса для результатов"""
        if self.weight_strategy == 'equal':
            return [1.0 / len(results)] * len(results)
        
        elif self.weight_strategy == 'quality':
            scores = [result.score for result in results]
            total_score = sum(scores)
            if total_score == 0:
                return [1.0 / len(results)] * len(results)
            return [score / total_score for score in scores]
        
        elif self.weight_strategy == 'frequency':
            # Вес на основе размера данных
            sizes = [len(result.data) if hasattr(result.data, '__len__') else 1 
                    for result in results]
            total_size = sum(sizes)
            if total_size == 0:
                return [1.0 / len(results)] * len(results)
            return [size / total_size for size in sizes]
        
        else:
            return [1.0 / len(results)] * len(results)
    
    def _analyze_results(self, results: List[ProcessingResult]) -> Dict[str, float]:
        """Анализирует характеристики результатов"""
        scores = [result.score for result in results]
        sizes = [len(result.data) if hasattr(result.data, '__len__') else 1 
                for result in results]
        
        import statistics
        
        analysis = {
            'avg_score': statistics.mean(scores),
            'quality_variance': statistics.variance(scores) if len(scores) > 1 else 0,
            'avg_size': statistics.mean(sizes),
            'size_variance': statistics.variance(sizes) if len(sizes) > 1 else 0,
            'count': len(results)
        }
        
        return analysis
    
    def _empty_result(self) -> ProcessingResult:
        """Создает пустой результат"""
        return ProcessingResult(
            data=[],
            stage=ProcessingStage.FUSION,
            modality=ModalityType.TEXT,
            metadata={"fusion_method": self.fusion_method, "empty": True},
            component_name=self.name
        )
    
    def _update_fusion_stats(self, input_data: List[ProcessingResult], 
                           output_result: ProcessingResult) -> None:
        """Обновляет статистику слияния"""
        self.fusion_stats['total_fusions'] += 1
        
        # Обновляем среднее количество входов
        total_fusions = self.fusion_stats['total_fusions']
        old_avg_input = self.fusion_stats['avg_input_count']
        self.fusion_stats['avg_input_count'] = (
            (old_avg_input * (total_fusions - 1) + len(input_data)) / total_fusions
        )
        
        # Обновляем средний размер выхода
        output_size = len(output_result.data) if hasattr(output_result.data, '__len__') else 1
        old_avg_output = self.fusion_stats['avg_output_size']
        self.fusion_stats['avg_output_size'] = (
            (old_avg_output * (total_fusions - 1) + output_size) / total_fusions
        )
        
        # Статистика по методам
        if self.fusion_method not in self.fusion_stats['method_usage']:
            self.fusion_stats['method_usage'][self.fusion_method] = 0
        self.fusion_stats['method_usage'][self.fusion_method] += 1
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """Возвращает статистику слияния"""
        stats = super().get_stats()
        stats.update({
            'fusion_stats': self.fusion_stats.copy(),
            'config': {
                'method': self.fusion_method,
                'weight_strategy': self.weight_strategy,
                'quality_threshold': self.quality_threshold,
                'max_results': self.max_results
            }
        })
        return stats
    
    def _get_default_stage(self) -> ProcessingStage:
        """Слияние - это этап слияния"""
        return ProcessingStage.FUSION


class ModelManager(StatefulComponent):
    """
    Управляет сохранением и загрузкой текстовых моделей
    
    Возможности:
    - Сохранение/загрузка состояния компонентов
    - Версионирование моделей
    - Экспорт в различные форматы
    - Статистика моделей
    - Автоматическое резервное копирование
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Инициализация менеджера моделей
        
        Args:
            name: Имя компонента
            config: Конфигурация менеджера
        """
        super().__init__(name, config)
        
        self.model_path = config.get('model_path', 'text_model.pkl')
        self.backup_path = config.get('backup_path', 'text_model_backup.pkl')
        self.auto_backup = config.get('auto_backup', True)
        self.backup_interval = config.get('backup_interval', 10)  # Каждые 10 операций
        self.compression = config.get('compression', True)
        self.versioning = config.get('versioning', True)
        
        # Зарегистрированные компоненты для сохранения
        self.components: Dict[str, Component] = {}
        
        # Статистика операций
        self.operation_stats = {
            'saves': 0,
            'loads': 0,
            'backups': 0,
            'exports': 0,
            'last_save_time': None,
            'last_load_time': None,
            'model_versions': [],
            'total_size_saved': 0
        }
        
        # Блокировка для многопоточности
        self._lock = threading.Lock()
        
        logger.debug(f"Инициализирован ModelManager {name} с путем {self.model_path}")
    
    def register_component(self, component_name: str, component: Component) -> None:
        """
        Регистрирует компонент для сохранения
        
        Args:
            component_name: Имя компонента
            component: Экземпляр компонента
        """
        with self._lock:
            self.components[component_name] = component
            logger.debug(f"Зарегистрирован компонент '{component_name}' для сохранения")
    
    def unregister_component(self, component_name: str) -> None:
        """Отменяет регистрацию компонента"""
        with self._lock:
            if component_name in self.components:
                del self.components[component_name]
                logger.debug(f"Отменена регистрация компонента '{component_name}'")
    
    def process(self, input_data: str) -> ProcessingResult:
        """
        Обработка команд управления моделью
        
        Args:
            input_data: Команда ('save', 'load', 'info', 'backup', 'export')
            
        Returns:
            ProcessingResult с результатом операции
        """
        command = input_data.lower().strip()
        
        try:
            if command == "save":
                return self._save_model()
            elif command == "load":
                return self._load_model()
            elif command == "info":
                return self._get_model_info()
            elif command == "backup":
                return self._backup_model()
            elif command.startswith("export"):
                # Поддержка команд типа "export json" или "export csv"
                parts = command.split()
                format_type = parts[1] if len(parts) > 1 else "json"
                return self._export_model(format_type)
            elif command == "cleanup":
                return self._cleanup_old_versions()
            else:
                raise ValueError(f"Неизвестная команда: {command}")
                
        except Exception as e:
            error_msg = f"Ошибка выполнения команды '{command}': {e}"
            logger.error(error_msg)
            
            return ProcessingResult(
                data=None,
                stage=ProcessingStage.GENERATION,
                modality=ModalityType.TEXT,
                error=error_msg,
                component_name=self.name
            )
    
    def _save_model(self) -> ProcessingResult:
        """Сохраняет состояние всех зарегистрированных компонентов"""
        with self._lock:
            model_data = {
                'version': self._get_next_version(),
                'timestamp': time.time(),
                'components': {},
                'metadata': {
                    'total_components': len(self.components),
                    'save_count': self.operation_stats['saves'] + 1
                }
            }
            
            # Собираем данные компонентов
            for comp_name, component in self.components.items():
                try:
                    comp_data = {
                        'type': component.__class__.__name__,
                        'config': getattr(component, 'config', {}),
                        'state': self._extract_component_state(component)
                    }
                    model_data['components'][comp_name] = comp_data
                except Exception as e:
                    logger.warning(f"Ошибка сохранения компонента {comp_name}: {e}")
                    model_data['components'][comp_name] = {
                        'type': component.__class__.__name__,
                        'error': str(e)
                    }
            
            try:
                # Создаем резервную копию если требуется
                if self.auto_backup and os.path.exists(self.model_path):
                    self._create_backup()
                
                # Сохраняем модель
                if self.compression:
                    self._save_compressed(model_data)
                else:
                    with open(self.model_path, 'wb') as f:
                        pickle.dump(model_data, f)
                
                # Обновляем статистику
                self.operation_stats['saves'] += 1
                self.operation_stats['last_save_time'] = time.time()
                
                if self.versioning:
                    self.operation_stats['model_versions'].append({
                        'version': model_data['version'],
                        'timestamp': model_data['timestamp'],
                        'components': len(model_data['components'])
                    })
                
                # Автоматический backup по интервалу
                if (self.auto_backup and 
                    self.operation_stats['saves'] % self.backup_interval == 0):
                    self._create_backup()
                
                result_msg = f"Модель сохранена: {self.model_path} (версия {model_data['version']})"
                
                return ProcessingResult(
                    data=result_msg,
                    stage=ProcessingStage.GENERATION,
                    modality=ModalityType.TEXT,
                    score=1.0,
                    metadata={
                        "operation": "save",
                        "saved_components": len(model_data['components']),
                        "version": model_data['version'],
                        "file_path": self.model_path
                    },
                    component_name=self.name
                )
                
            except Exception as e:
                error_msg = f"Ошибка сохранения модели: {e}"
                return ProcessingResult(
                    data=error_msg,
                    stage=ProcessingStage.GENERATION,
                    modality=ModalityType.TEXT,
                    error=error_msg,
                    component_name=self.name
                )
    
    def _load_model(self) -> ProcessingResult:
        """Загружает состояние компонентов из файла"""
        if not os.path.exists(self.model_path):
            error_msg = f"Файл модели не найден: {self.model_path}"
            return ProcessingResult(
                data=error_msg,
                stage=ProcessingStage.GENERATION,
                modality=ModalityType.TEXT,
                error=error_msg,
                component_name=self.name
            )
        
        try:
            with self._lock:
                # Загружаем данные
                if self.compression:
                    model_data = self._load_compressed()
                else:
                    with open(self.model_path, 'rb') as f:
                        model_data = pickle.load(f)
                
                loaded_count = 0
                errors = []
                
                # Восстанавливаем состояние компонентов
                for comp_name, comp_data in model_data.get('components', {}).items():
                    if comp_name in self.components:
                        try:
                            self._restore_component_state(self.components[comp_name], comp_data)
                            loaded_count += 1
                        except Exception as e:
                            error_msg = f"Ошибка загрузки компонента {comp_name}: {e}"
                            errors.append(error_msg)
                            logger.warning(error_msg)
                
                # Обновляем статистику
                self.operation_stats['loads'] += 1
                self.operation_stats['last_load_time'] = time.time()
                
                result_msg = f"Модель загружена: {loaded_count} компонентов"
                if errors:
                    result_msg += f", {len(errors)} ошибок"
                
                return ProcessingResult(
                    data=result_msg,
                    stage=ProcessingStage.GENERATION,
                    modality=ModalityType.TEXT,
                    score=1.0 if not errors else 0.8,
                    metadata={
                        "operation": "load",
                        "loaded_components": loaded_count,
                        "errors": errors,
                        "version": model_data.get('version', 'unknown'),
                        "timestamp": model_data.get('timestamp')
                    },
                    component_name=self.name
                )
                
        except Exception as e:
            error_msg = f"Ошибка загрузки модели: {e}"
            return ProcessingResult(
                data=error_msg,
                stage=ProcessingStage.GENERATION,
                modality=ModalityType.TEXT,
                error=error_msg,
                component_name=self.name
            )
    
    def _get_model_info(self) -> ProcessingResult:
        """Возвращает информацию о модели и зарегистрированных компонентах"""
        with self._lock:
            info = {
                "registered_components": len(self.components),
                "model_path": self.model_path,
                "model_exists": os.path.exists(self.model_path),
                "operations": self.operation_stats.copy(),
                "components": {
                    name: {
                        "type": comp.__class__.__name__,
                        "config": getattr(comp, 'config', {}),
                        "has_state": hasattr(comp, 'get_full_state')
                    }
                    for name, comp in self.components.items()
                }
            }
            
            if os.path.exists(self.model_path):
                try:
                    stat = os.stat(self.model_path)
                    info["file_info"] = {
                        "size_bytes": stat.st_size,
                        "modified_time": stat.st_mtime
                    }
                except Exception as e:
                    info["file_info"] = {"error": str(e)}
        
        return ProcessingResult(
            data=info,
            stage=ProcessingStage.GENERATION,
            modality=ModalityType.TEXT,
            score=1.0,
            metadata={"operation": "info"},
            component_name=self.name
        )
    
    def _backup_model(self) -> ProcessingResult:
        """Создает резервную копию модели"""
        if not os.path.exists(self.model_path):
            error_msg = "Нет модели для резервного копирования"
            return ProcessingResult(
                data=error_msg,
                stage=ProcessingStage.GENERATION,
                modality=ModalityType.TEXT,
                error=error_msg,
                component_name=self.name
            )
        
        try:
            backup_created = self._create_backup()
            
            if backup_created:
                result_msg = f"Резервная копия создана: {self.backup_path}"
                return ProcessingResult(
                    data=result_msg,
                    stage=ProcessingStage.GENERATION,
                    modality=ModalityType.TEXT,
                    score=1.0,
                    metadata={"operation": "backup", "backup_path": self.backup_path},
                    component_name=self.name
                )
            else:
                error_msg = "Не удалось создать резервную копию"
                return ProcessingResult(
                    data=error_msg,
                    stage=ProcessingStage.GENERATION,
                    modality=ModalityType.TEXT,
                    error=error_msg,
                    component_name=self.name
                )
                
        except Exception as e:
            error_msg = f"Ошибка создания резервной копии: {e}"
            return ProcessingResult(
                data=error_msg,
                stage=ProcessingStage.GENERATION,
                modality=ModalityType.TEXT,
                error=error_msg,
                component_name=self.name
            )
    
    def _export_model(self, format_type: str) -> ProcessingResult:
        """Экспортирует модель в указанном формате"""
        if not os.path.exists(self.model_path):
            error_msg = "Нет модели для экспорта"
            return ProcessingResult(
                data=error_msg,
                stage=ProcessingStage.GENERATION,
                modality=ModalityType.TEXT,
                error=error_msg,
                component_name=self.name
            )
        
        try:
            # Определяем путь экспорта
            base_name = os.path.splitext(self.model_path)[0]
            export_path = f"{base_name}_export.{format_type}"
            
            if format_type == "json":
                exported = self._export_to_json(export_path)
            elif format_type == "csv":
                exported = self._export_to_csv(export_path)
            elif format_type == "txt":
                exported = self._export_to_text(export_path)
            else:
                raise ValueError(f"Неподдерживаемый формат экспорта: {format_type}")
            
            if exported:
                self.operation_stats['exports'] += 1
                result_msg = f"Модель экспортирована в {format_type}: {export_path}"
                
                return ProcessingResult(
                    data=result_msg,
                    stage=ProcessingStage.GENERATION,
                    modality=ModalityType.TEXT,
                    score=1.0,
                    metadata={
                        "operation": "export",
                        "format": format_type,
                        "export_path": export_path
                    },
                    component_name=self.name
                )
            else:
                error_msg = f"Не удалось экспортировать в формат {format_type}"
                return ProcessingResult(
                    data=error_msg,
                    stage=ProcessingStage.GENERATION,
                    modality=ModalityType.TEXT,
                    error=error_msg,
                    component_name=self.name
                )
                
        except Exception as e:
            error_msg = f"Ошибка экспорта: {e}"
            return ProcessingResult(
                data=error_msg,
                stage=ProcessingStage.GENERATION,
                modality=ModalityType.TEXT,
                error=error_msg,
                component_name=self.name
            )
    
    def _extract_component_state(self, component: Component) -> Dict[str, Any]:
        """Извлекает состояние компонента для сохранения"""
        state = {}
        
        # Базовые атрибуты
        for attr in ['name', 'config']:
            if hasattr(component, attr):
                state[attr] = getattr(component, attr)
        
        # Специальное состояние для StatefulComponent
        if hasattr(component, 'get_full_state'):
            state['stateful_data'] = component.get_full_state()
        
        # Статистика если есть
        if hasattr(component, 'get_stats'):
            state['stats'] = component.get_stats()
        
        # Специфичные для типа данные
        component_type = component.__class__.__name__
        
        if component_type == 'NGramExtractor':
            state['ngram_data'] = {
                'forward_graphs': getattr(component, 'forward_graphs', {}),
                'backward_graphs': getattr(component, 'backward_graphs', {}),
                'counts': getattr(component, 'counts', {}),
                'totals': getattr(component, 'totals', {}),
                'adaptive_weights': getattr(component, 'adaptive_weights', {})
            }
        
        elif component_type == 'AdvancedTextTokenizer':
            state['tokenizer_data'] = {
                'stats': getattr(component, 'stats', {}),
                'special_tokens': getattr(component, 'special_tokens', {})
            }
        
        return state
    
    def _restore_component_state(self, component: Component, comp_data: Dict[str, Any]) -> None:
        """Восстанавливает состояние компонента"""
        state = comp_data.get('state', {})
        
        # Восстанавливаем базовые атрибуты
        for attr, value in state.items():
            if attr in ['name', 'config'] and hasattr(component, attr):
                setattr(component, attr, value)
        
        # Восстанавливаем stateful данные
        if 'stateful_data' in state and hasattr(component, 'update_state'):
            component.update_state(state['stateful_data'])
        
        # Восстанавливаем специфичные данные
        component_type = component.__class__.__name__
        
        if component_type == 'NGramExtractor' and 'ngram_data' in state:
            ngram_data = state['ngram_data']
            for attr in ['forward_graphs', 'backward_graphs', 'counts', 'totals', 'adaptive_weights']:
                if attr in ngram_data and hasattr(component, attr):
                    setattr(component, attr, ngram_data[attr])
        
        elif component_type == 'AdvancedTextTokenizer' and 'tokenizer_data' in state:
            tokenizer_data = state['tokenizer_data']
            for attr in ['stats', 'special_tokens']:
                if attr in tokenizer_data and hasattr(component, attr):
                    setattr(component, attr, tokenizer_data[attr])
    
    def _get_next_version(self) -> str:
        """Генерирует следующую версию модели"""
        if not self.versioning:
            return "1.0"
        
        versions = self.operation_stats.get('model_versions', [])
        if not versions:
            return "1.0"
        
        # Простое инкрементирование версии
        last_version = versions[-1]['version']
        try:
            major, minor = map(int, last_version.split('.'))
            return f"{major}.{minor + 1}"
        except:
            return f"1.{len(versions) + 1}"
    
    def _create_backup(self) -> bool:
        """Создает резервную копию модели"""
        try:
            import shutil
            if os.path.exists(self.model_path):
                # Добавляем timestamp к backup
                timestamp = int(time.time())
                backup_with_timestamp = f"{os.path.splitext(self.backup_path)[0]}_{timestamp}.pkl"
                
                shutil.copy2(self.model_path, backup_with_timestamp)
                # Также создаем основной backup без timestamp
                shutil.copy2(self.model_path, self.backup_path)
                
                self.operation_stats['backups'] += 1
                logger.info(f"Создана резервная копия: {backup_with_timestamp}")
                return True
        except Exception as e:
            logger.error(f"Ошибка создания резервной копии: {e}")
        
        return False
    
    def _save_compressed(self, data: Any) -> None:
        """Сохраняет данные со сжатием"""
        import gzip
        
        with gzip.open(f"{self.model_path}.gz", 'wb') as f:
            pickle.dump(data, f)
        
        # Перемещаем сжатый файл
        os.rename(f"{self.model_path}.gz", self.model_path)
    
    def _load_compressed(self) -> Any:
        """Загружает сжатые данные"""
        import gzip
        
        try:
            with gzip.open(self.model_path, 'rb') as f:
                return pickle.load(f)
        except:
            # Fallback на обычную загрузку
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)
    
    def _export_to_json(self, export_path: str) -> bool:
        """Экспортирует модель в JSON"""
        try:
            import json
            
            # Создаем упрощенную версию для JSON
            export_data = {
                "model_info": self._get_model_info().data,
                "components_summary": {
                    name: {
                        "type": comp.__class__.__name__,
                        "config": getattr(comp, 'config', {})
                    }
                    for name, comp in self.components.items()
                }
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            return True
        except Exception as e:
            logger.error(f"Ошибка экспорта в JSON: {e}")
            return False
    
    def _export_to_csv(self, export_path: str) -> bool:
        """Экспортирует статистику модели в CSV"""
        try:
            import csv
            
            with open(export_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Заголовки
                writer.writerow(['Component', 'Type', 'Config_Keys', 'Stats_Available'])
                
                # Данные компонентов
                for name, comp in self.components.items():
                    config_keys = list(getattr(comp, 'config', {}).keys())
                    has_stats = hasattr(comp, 'get_stats')
                    
                    writer.writerow([
                        name,
                        comp.__class__.__name__,
                        ';'.join(config_keys),
                        has_stats
                    ])
            
            return True
        except Exception as e:
            logger.error(f"Ошибка экспорта в CSV: {e}")
            return False
    
    def _export_to_text(self, export_path: str) -> bool:
        """Экспортирует информацию о модели в текстовый файл"""
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write("ARIA Model Export\n")
                f.write("=" * 50 + "\n\n")
                
                # Общая информация
                f.write(f"Model Path: {self.model_path}\n")
                f.write(f"Components: {len(self.components)}\n")
                f.write(f"Operations: {self.operation_stats}\n\n")
                
                # Детали компонентов
                f.write("Registered Components:\n")
                f.write("-" * 30 + "\n")
                
                for name, comp in self.components.items():
                    f.write(f"\nComponent: {name}\n")
                    f.write(f"  Type: {comp.__class__.__name__}\n")
                    f.write(f"  Config: {getattr(comp, 'config', {})}\n")
                    
                    if hasattr(comp, 'get_stats'):
                        stats = comp.get_stats()
                        f.write(f"  Stats: {stats}\n")
            
            return True
        except Exception as e:
            logger.error(f"Ошибка экспорта в текст: {e}")
            return False
    
    def _cleanup_old_versions(self) -> ProcessingResult:
        """Очищает старые версии моделей"""
        # Простая реализация - можно расширить
        cleaned = 0
        try:
            # Находим файлы резервных копий
            import glob
            backup_pattern = f"{os.path.splitext(self.backup_path)[0]}_*.pkl"
            backup_files = glob.glob(backup_pattern)
            
            # Оставляем только последние 5 версий
            if len(backup_files) > 5:
                backup_files.sort()
                to_remove = backup_files[:-5]
                
                for file_path in to_remove:
                    try:
                        os.remove(file_path)
                        cleaned += 1
                    except Exception as e:
                        logger.warning(f"Не удалось удалить {file_path}: {e}")
        
        except Exception as e:
            logger.error(f"Ошибка очистки версий: {e}")
        
        result_msg = f"Очищено старых версий: {cleaned}"
        return ProcessingResult(
            data=result_msg,
            stage=ProcessingStage.GENERATION,
            modality=ModalityType.TEXT,
            score=1.0,
            metadata={"operation": "cleanup", "cleaned_count": cleaned},
            component_name=self.name
        )
    
    def _get_default_stage(self) -> ProcessingStage:
        """Управление моделями - это генерация результатов"""
        return ProcessingStage.GENERATION