# aria/processors/text/ngrams.py
"""
Извлечение и обработка n-грамм для ARIA
"""

import math
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
import threading
import logging

from ...core import StatefulComponent, ModalityType, ProcessingResult, ProcessingStage

logger = logging.getLogger(__name__)


@dataclass
class NGramConfig:
    """Конфигурация для извлечения n-грамм"""
    min_n: int = 2
    max_n: int = 5
    smoothing_factor: float = 0.1
    dynamic_selection: bool = True
    interpolation_method: str = 'adaptive'  # 'adaptive', 'fixed', 'learned'
    min_frequency: int = 2
    max_ngrams: Optional[int] = None  # Ограничение на количество n-грамм
    
    def __post_init__(self):
        if self.min_n < 1:
            raise ValueError("min_n должно быть >= 1")
        if self.max_n < self.min_n:
            raise ValueError("max_n должно быть >= min_n")
        if self.smoothing_factor < 0:
            raise ValueError("smoothing_factor должен быть >= 0")


@dataclass
class BeamCandidate:
    """Кандидат для beam search"""
    tokens: List[str]
    score: float
    normalized_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        return self.normalized_score < other.normalized_score


class NGramExtractor(StatefulComponent):
    """
    Извлекатель переменных n-грамм с адаптивными весами
    
    Возможности:
    - Извлечение n-грамм разных уровней (2-5)
    - Адаптивные веса на основе плотности данных
    - Построение графов переходов (прямых и обратных)
    - Интерполяция вероятностей между уровнями
    - Динамический выбор оптимального размера n-граммы
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Инициализация извлекателя n-грамм
        
        Args:
            name: Имя компонента
            config: Конфигурация
        """
        super().__init__(name, config)
        
        # Парсинг конфигурации
        ngram_config = config.get('ngram', {})
        self.ngram_config = NGramConfig(**ngram_config)
        
        # Структуры данных для каждого уровня n
        self.forward_graphs = {}
        self.backward_graphs = {}
        self.counts = {}
        self.totals = {}
        
        # Инициализация для всех уровней n
        for n in range(self.ngram_config.min_n, self.ngram_config.max_n + 1):
            self.forward_graphs[n] = defaultdict(dict)
            self.backward_graphs[n] = defaultdict(dict)
            self.counts[n] = defaultdict(int)
            self.totals[n] = 0
        
        # Адаптивные веса для интерполяции
        self.adaptive_weights = self._initialize_weights()
        self.weight_history = []
        
        # Статистика
        self.extraction_stats = {
            'total_sentences': 0,
            'total_ngrams_extracted': 0,
            'level_usage': {n: 0 for n in range(self.ngram_config.min_n, self.ngram_config.max_n + 1)},
            'unique_ngrams_per_level': {n: 0 for n in range(self.ngram_config.min_n, self.ngram_config.max_n + 1)},
            'weight_updates': 0
        }
        
        logger.debug(f"Инициализирован NGramExtractor {name} с конфигурацией: {self.ngram_config}")
    
    def process(self, input_data: List[str]) -> ProcessingResult:
        """
        Извлекает n-граммы из токенов
        
        Args:
            input_data: Список токенов
            
        Returns:
            ProcessingResult с извлеченными n-граммами
        """
        if not isinstance(input_data, list):
            raise ValueError("Входные данные должны быть списком токенов")
        
        if len(input_data) < self.ngram_config.min_n:
            logger.warning(f"Недостаточно токенов для извлечения {self.ngram_config.min_n}-грамм: {len(input_data)}")
            return ProcessingResult(
                data={},
                stage=ProcessingStage.ENCODING,
                modality=ModalityType.TEXT,
                metadata={"warning": "insufficient_tokens"},
                component_name=self.name
            )
        
        # Извлекаем n-граммы и обновляем модель
        ngrams_data = self._extract_and_update_ngrams(input_data)
        
        # Обновляем адаптивные веса
        self._update_adaptive_weights()
        
        # Обновляем статистику
        self._update_extraction_stats(ngrams_data)
        
        return ProcessingResult(
            data=ngrams_data,
            stage=ProcessingStage.ENCODING,
            modality=ModalityType.TEXT,
            metadata={
                "total_ngrams": sum(len(ngrams) for ngrams in ngrams_data.values()),
                "levels_processed": list(ngrams_data.keys()),
                "adaptive_weights": self.adaptive_weights.copy(),
                "tokens_processed": len(input_data),
                "ngram_config": {
                    "min_n": self.ngram_config.min_n,
                    "max_n": self.ngram_config.max_n,
                    "dynamic_selection": self.ngram_config.dynamic_selection
                }
            },
            component_name=self.name
        )
    
    def _initialize_weights(self) -> Dict[int, float]:
        """Инициализирует веса для интерполяции"""
        total_levels = self.ngram_config.max_n - self.ngram_config.min_n + 1
        return {n: 1.0 / total_levels for n in range(self.ngram_config.min_n, self.ngram_config.max_n + 1)}
    
    def _extract_and_update_ngrams(self, tokens: List[str]) -> Dict[int, List[Tuple[str, ...]]]:
        """
        Извлекает n-граммы всех уровней и обновляет модель
        
        Args:
            tokens: Список токенов
            
        Returns:
            Словарь n-грамм по уровням
        """
        ngrams_by_level = {}
        
        for n in range(self.ngram_config.min_n, 
                       min(self.ngram_config.max_n + 1, len(tokens) + 1)):
            ngrams = []
            
            # Извлекаем n-граммы уровня n
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                ngrams.append(ngram)
                
                # Обновляем модель
                self._update_ngram_model(ngram, tokens, i, n)
            
            ngrams_by_level[n] = ngrams
            self.extraction_stats['level_usage'][n] += len(ngrams)
        
        return ngrams_by_level
    
    def _update_ngram_model(self, ngram: Tuple[str, ...], tokens: List[str], pos: int, n: int) -> None:
        """
        Обновляет модель n-грамм с переходами
        
        Args:
            ngram: N-грамма для обновления
            tokens: Исходные токены
            pos: Позиция n-граммы в тексте
            n: Размер n-граммы
        """
        # Обновляем счетчики
        self.counts[n][ngram] += 1
        self.totals[n] += 1
        
        # Прямое направление (следующий токен)
        if pos + n < len(tokens):
            next_token = tokens[pos + n]
            if ngram not in self.forward_graphs[n]:
                self.forward_graphs[n][ngram] = {}
            self.forward_graphs[n][ngram][next_token] = \
                self.forward_graphs[n][ngram].get(next_token, 0) + 1
        
        # Обратное направление (предыдущий токен)
        if pos > 0:
            prev_token = tokens[pos - 1]
            if ngram not in self.backward_graphs[n]:
                self.backward_graphs[n][ngram] = {}
            self.backward_graphs[n][ngram][prev_token] = \
                self.backward_graphs[n][ngram].get(prev_token, 0) + 1
    
    def _update_adaptive_weights(self) -> None:
        """Обновляет адаптивные веса на основе доступности данных"""
        if not self.ngram_config.dynamic_selection:
            return
        
        # Вычисляем плотность данных для каждого уровня
        densities = {}
        total_density = 0
        
        for n in range(self.ngram_config.min_n, self.ngram_config.max_n + 1):
            if self.totals[n] > 0:
                unique_ngrams = len(self.counts[n])
                avg_frequency = self.totals[n] / max(unique_ngrams, 1)
                coverage = min(unique_ngrams / 1000, 1.0)  # Нормализуем покрытие
                transition_density = len(self.forward_graphs[n]) / max(unique_ngrams, 1)
                
                # Комбинируем метрики
                densities[n] = avg_frequency * coverage * transition_density
                total_density += densities[n]
        
        # Обновляем веса с сглаживанием
        if total_density > 0:
            alpha = 0.1  # Фактор сглаживания
            for n in range(self.ngram_config.min_n, self.ngram_config.max_n + 1):
                new_weight = densities.get(n, 0) / total_density
                old_weight = self.adaptive_weights[n]
                self.adaptive_weights[n] = (1 - alpha) * old_weight + alpha * new_weight
            
            # Нормализуем веса
            total_weight = sum(self.adaptive_weights.values())
            if total_weight > 0:
                for n in self.adaptive_weights:
                    self.adaptive_weights[n] /= total_weight
            
            # Сохраняем историю
            self.weight_history.append(self.adaptive_weights.copy())
            if len(self.weight_history) > 100:  # Ограничиваем размер истории
                self.weight_history.pop(0)
            
            self.extraction_stats['weight_updates'] += 1
    
    def get_interpolated_probabilities(self, context: List[str]) -> Dict[str, float]:
        """
        Вычисляет интерполированные вероятности для всех возможных следующих токенов
        
        Args:
            context: Контекст (последние токены)
            
        Returns:
            Словарь {токен: вероятность}
        """
        all_candidates = set()
        level_probabilities = {}
        
        # Собираем кандидатов со всех уровней
        for n in range(self.ngram_config.min_n, self.ngram_config.max_n + 1):
            level_probabilities[n] = {}
            
            if len(context) >= n - 1:
                context_ngram = tuple(context[-(n-1):]) if n > 1 else ()
                
                # Получаем переходы
                transitions = self.forward_graphs[n].get(context_ngram, {})
                
                if transitions:
                    total_count = sum(transitions.values())
                    
                    for token, count in transitions.items():
                        # Применяем сглаживание
                        smoothed_count = count + self.ngram_config.smoothing_factor
                        smoothed_total = total_count + self.ngram_config.smoothing_factor * len(transitions)
                        
                        prob = smoothed_count / smoothed_total
                        level_probabilities[n][token] = prob
                        all_candidates.add(token)
        
        # Интерполируем вероятности
        interpolated = {}
        
        for token in all_candidates:
            interpolated_prob = 0
            
            for n in range(self.ngram_config.min_n, self.ngram_config.max_n + 1):
                weight = self.adaptive_weights.get(n, 0)
                prob = level_probabilities[n].get(token, 0)
                
                # Добавляем базовое сглаживание для отсутствующих токенов
                if prob == 0 and len(all_candidates) > 0:
                    prob = self.ngram_config.smoothing_factor / (len(all_candidates) * 10)
                
                interpolated_prob += weight * prob
            
            interpolated[token] = interpolated_prob
        
        # Нормализуем вероятности
        total_prob = sum(interpolated.values())
        if total_prob > 0:
            for token in interpolated:
                interpolated[token] /= total_prob
        
        return interpolated
    
    def select_optimal_n(self, context: List[str]) -> int:
        """
        Выбирает оптимальный размер n-граммы для данного контекста
        
        Args:
            context: Контекст для анализа
            
        Returns:
            Оптимальный размер n-граммы
        """
        if not self.ngram_config.dynamic_selection:
            return self.ngram_config.max_n
        
        scores = {}
        context_len = len(context)
        
        for n in range(self.ngram_config.min_n, min(self.ngram_config.max_n + 1, context_len + 2)):
            score = 0
            
            # Базовый вес уровня
            score += self.adaptive_weights.get(n, 0) * 100
            
            # Бонус за доступность контекста
            if context_len >= n - 1:
                context_ngram = tuple(context[-(n-1):]) if n > 1 else ()
                
                # Проверяем доступность переходов
                available_transitions = len(self.forward_graphs[n].get(context_ngram, {}))
                
                # Добавляем бонус за наличие переходов
                score += min(available_transitions * 10, 100)
                
                # Особый бонус если есть хотя бы один переход
                if available_transitions > 0:
                    score += 50
                
                # Бонус за частоту использования n-граммы
                ngram_count = self.counts[n].get(context_ngram, 0)
                if ngram_count > 0:
                    score += min(math.log(ngram_count + 1) * 10, 50)
            
            # Штраф за недостаток контекста
            if context_len < n - 1:
                score *= 0.2
            
            scores[n] = score
        
        # Возвращаем уровень с наивысшим счетом
        if scores:
            best_n = max(scores.items(), key=lambda x: x[1])[0]
            logger.debug(f"Выбран оптимальный n={best_n} для контекста {context[-3:]} (счета: {scores})")
            return best_n
        else:
            return self.ngram_config.min_n
    
    def _update_extraction_stats(self, ngrams_data: Dict[int, List[Tuple[str, ...]]]) -> None:
        """Обновляет статистику извлечения"""
        self.extraction_stats['total_sentences'] += 1
        total_ngrams = sum(len(ngrams) for ngrams in ngrams_data.values())
        self.extraction_stats['total_ngrams_extracted'] += total_ngrams
        
        # Обновляем количество уникальных n-грамм
        for n, ngrams in ngrams_data.items():
            unique_ngrams = len(set(ngrams))
            self.extraction_stats['unique_ngrams_per_level'][n] = len(self.counts[n])
    
    def get_ngram_statistics(self) -> Dict[str, Any]:
        """
        Возвращает детальную статистику n-грамм
        
        Returns:
            Словарь со статистикой
        """
        stats = {
            'extraction_stats': self.extraction_stats.copy(),
            'adaptive_weights': self.adaptive_weights.copy(),
            'model_stats': {}
        }
        
        # Статистика по уровням
        for n in range(self.ngram_config.min_n, self.ngram_config.max_n + 1):
            unique_ngrams = len(self.counts[n])
            total_ngrams = self.totals[n]
            forward_transitions = len(self.forward_graphs[n])
            backward_transitions = len(self.backward_graphs[n])
            
            avg_frequency = total_ngrams / max(unique_ngrams, 1)
            coverage_ratio = forward_transitions / max(unique_ngrams, 1)
            
            stats['model_stats'][f'level_{n}'] = {
                'unique_ngrams': unique_ngrams,
                'total_ngrams': total_ngrams,
                'avg_frequency': avg_frequency,
                'forward_transitions': forward_transitions,
                'backward_transitions': backward_transitions,
                'coverage_ratio': coverage_ratio,
                'weight': self.adaptive_weights.get(n, 0)
            }
        
        return stats
    
    def get_most_frequent_ngrams(self, n: int, top_k: int = 10) -> List[Tuple[Tuple[str, ...], int]]:
        """
        Возвращает наиболее частые n-граммы уровня n
        
        Args:
            n: Размер n-граммы
            top_k: Количество топ n-грамм
            
        Returns:
            Список (n-грамма, частота)
        """
        if n not in self.counts:
            return []
        
        # Сортируем по частоте
        sorted_ngrams = sorted(self.counts[n].items(), key=lambda x: x[1], reverse=True)
        return sorted_ngrams[:top_k]
    
    def get_ngram_transitions(self, ngram: Tuple[str, ...], direction: str = 'forward') -> Dict[str, float]:
        """
        Возвращает переходы для конкретной n-граммы
        
        Args:
            ngram: N-грамма
            direction: 'forward' или 'backward'
            
        Returns:
            Словарь переходов с вероятностями
        """
        n = len(ngram)
        if n not in self.forward_graphs:
            return {}
        
        if direction == 'forward':
            transitions = self.forward_graphs[n].get(ngram, {})
        elif direction == 'backward':
            transitions = self.backward_graphs[n].get(ngram, {})
        else:
            raise ValueError("direction должно быть 'forward' или 'backward'")
        
        if not transitions:
            return {}
        
        # Преобразуем в вероятности
        total_count = sum(transitions.values())
        probabilities = {token: count / total_count for token, count in transitions.items()}
        
        return probabilities
    
    def prune_ngrams(self, min_frequency: Optional[int] = None) -> Dict[str, int]:
        """
        Удаляет редкие n-граммы для экономии памяти
        
        Args:
            min_frequency: Минимальная частота (по умолчанию из конфигурации)
            
        Returns:
            Статистика удаления {уровень: количество_удаленных}
        """
        min_freq = min_frequency or self.ngram_config.min_frequency
        removal_stats = {}
        
        for n in range(self.ngram_config.min_n, self.ngram_config.max_n + 1):
            removed_count = 0
            
            # Находим n-граммы для удаления
            to_remove = [ngram for ngram, count in self.counts[n].items() 
                        if count < min_freq]
            
            # Удаляем n-граммы
            for ngram in to_remove:
                # Удаляем из счетчиков
                removed_count += self.counts[n][ngram]
                del self.counts[n][ngram]
                
                # Удаляем из графов переходов
                if ngram in self.forward_graphs[n]:
                    del self.forward_graphs[n][ngram]
                if ngram in self.backward_graphs[n]:
                    del self.backward_graphs[n][ngram]
            
            self.totals[n] -= removed_count
            removal_stats[f'level_{n}'] = len(to_remove)
        
        logger.info(f"Удалено редких n-грамм: {removal_stats}")
        return removal_stats
    
    def save_model(self, file_path: str) -> None:
        """
        Сохраняет модель n-грамм в файл
        
        Args:
            file_path: Путь для сохранения
        """
        model_data = {
            'config': self.ngram_config,
            'forward_graphs': dict(self.forward_graphs),
            'backward_graphs': dict(self.backward_graphs),
            'counts': dict(self.counts),
            'totals': dict(self.totals),
            'adaptive_weights': self.adaptive_weights,
            'extraction_stats': self.extraction_stats,
            'weight_history': self.weight_history
        }
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Модель n-грамм сохранена: {file_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
            raise
    
    def load_model(self, file_path: str) -> None:
        """
        Загружает модель n-грамм из файла
        
        Args:
            file_path: Путь к файлу модели
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Восстанавливаем состояние
            self.ngram_config = model_data.get('config', self.ngram_config)
            
            # Восстанавливаем структуры данных
            for n in range(self.ngram_config.min_n, self.ngram_config.max_n + 1):
                self.forward_graphs[n] = defaultdict(dict)
                self.backward_graphs[n] = defaultdict(dict)
                self.counts[n] = defaultdict(int)
                
                # Загружаем данные если есть
                if 'forward_graphs' in model_data and n in model_data['forward_graphs']:
                    for ngram, transitions in model_data['forward_graphs'][n].items():
                        self.forward_graphs[n][ngram] = dict(transitions)
                
                if 'backward_graphs' in model_data and n in model_data['backward_graphs']:
                    for ngram, transitions in model_data['backward_graphs'][n].items():
                        self.backward_graphs[n][ngram] = dict(transitions)
                
                if 'counts' in model_data and n in model_data['counts']:
                    for ngram, count in model_data['counts'][n].items():
                        self.counts[n][ngram] = count
            
            self.totals = model_data.get('totals', {})
            self.adaptive_weights = model_data.get('adaptive_weights', self._initialize_weights())
            self.extraction_stats = model_data.get('extraction_stats', self.extraction_stats)
            self.weight_history = model_data.get('weight_history', [])
            
            logger.info(f"Модель n-грамм загружена: {file_path}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def _get_default_stage(self) -> ProcessingStage:
        """N-gram extraction - это этап кодирования"""
        return ProcessingStage.ENCODING