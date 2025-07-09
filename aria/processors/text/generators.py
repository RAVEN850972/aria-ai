# aria/processors/text/generators.py
"""
Генераторы текста для ARIA
"""

import math
import random
import time
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from ...core import Component, ModalityType, ProcessingResult, ProcessingStage
from .ngrams import NGramExtractor, BeamCandidate

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
   """Конфигурация генерации текста"""
   # Beam Search параметры
   beam_size: int = 5
   length_penalty: float = 0.6
   repetition_penalty: float = 1.2
   diversity_penalty: float = 0.5
   
   # Nucleus Sampling параметры  
   top_p: float = 0.9
   temperature: float = 0.8
   min_tokens_to_keep: int = 1
   
   # Параметры длины
   max_length: int = 20
   min_length: int = 3
   early_stopping: bool = True
   
   # Многопоточность
   max_workers: int = 4
   parallel_beam_search: bool = True
   
   # Стратегии генерации
   generation_strategy: str = "beam_search"  # "beam_search", "nucleus", "greedy"
   fallback_strategy: str = "nucleus"
   
   def __post_init__(self):
       if self.beam_size < 1:
           raise ValueError("beam_size должен быть >= 1")
       if not 0.0 <= self.top_p <= 1.0:
           raise ValueError("top_p должен быть в диапазоне [0.0, 1.0]")
       if self.temperature <= 0:
           raise ValueError("temperature должен быть > 0")
       if self.max_length < self.min_length:
           raise ValueError("max_length должен быть >= min_length")


class BeamSearchGenerator(Component):
   """
   Генератор текста с использованием beam search и nucleus sampling
   
   Возможности:
   - Beam search с различными стратегиями
   - Nucleus (top-p) sampling
   - Параллельная обработка кандидатов
   - Штрафы за повторения и длину
   - Динамическое переключение стратегий
   """
   
   def __init__(self, name: str, config: Dict[str, Any] = None):
       """
       Инициализация генератора
       
       Args:
           name: Имя компонента
           config: Конфигурация генератора
       """
       super().__init__(name, config)
       
       # Парсинг конфигурации
       gen_config = config.get('generation', {})
       self.gen_config = GenerationConfig(**gen_config)
       
       # Связанная модель n-грамм
       self.ngram_extractor: Optional[NGramExtractor] = None
       
       # Статистика генерации
       self.generation_stats = {
           'total_generations': 0,
           'beam_search_usage': 0,
           'nucleus_sampling_usage': 0,
           'parallel_generations': 0,
           'avg_generation_time': 0.0,
           'avg_beam_time': 0.0,
           'strategy_usage': {
               'beam_search': 0,
               'nucleus': 0,
               'greedy': 0,
               'fallback': 0
           },
           'length_stats': {
               'avg_generated_length': 0.0,
               'min_generated': float('inf'),
               'max_generated': 0
           }
       }
       
       logger.debug(f"Инициализирован BeamSearchGenerator {name} с конфигурацией: {self.gen_config}")
   
   def set_ngram_model(self, ngram_extractor: NGramExtractor) -> None:
       """
       Устанавливает модель n-грамм для генерации
       
       Args:
           ngram_extractor: Извлекатель n-грамм
       """
       self.ngram_extractor = ngram_extractor
       logger.debug(f"Установлена модель n-грамм для генератора {self.name}")
   
   def process(self, input_data: Any) -> ProcessingResult:
       """
       Генерирует текст на основе входных токенов
       
       Args:
           input_data: Начальные токены для генерации (список или ModalityData)
           
       Returns:
           ProcessingResult с сгенерированным текстом
       """
       # Поддерживаем разные форматы входных данных
       if hasattr(input_data, 'data'):
           # Это ModalityData из другого компонента
           tokens = input_data.data
       elif isinstance(input_data, dict) and len(input_data) == 1:
           # Это единственный вход из графа
           tokens = next(iter(input_data.values()))
           if hasattr(tokens, 'data'):
               tokens = tokens.data
       else:
           tokens = input_data
       
       if not isinstance(tokens, list):
           raise ValueError("Токены должны быть списком строк")
       
       if not self.ngram_extractor:
           raise ValueError("N-gram модель не установлена. Используйте set_ngram_model()")
       
       start_time = time.time()
       
       # Выбираем стратегию генерации
       strategy = self._select_generation_strategy(tokens)
       
       try:
           # Генерируем текст
           if strategy == "beam_search":
               generated_tokens, metadata = self._beam_search_generate(tokens)
               self.generation_stats['beam_search_usage'] += 1
           elif strategy == "nucleus":
               generated_tokens, metadata = self._nucleus_generate(tokens)
               self.generation_stats['nucleus_sampling_usage'] += 1
           elif strategy == "greedy":
               generated_tokens, metadata = self._greedy_generate(tokens)
           else:
               # Fallback
               generated_tokens, metadata = self._fallback_generate(tokens)
               self.generation_stats['strategy_usage']['fallback'] += 1
           
           self.generation_stats['strategy_usage'][strategy] += 1
           
       except Exception as e:
           logger.warning(f"Ошибка в стратегии {strategy}, переключение на fallback: {e}")
           generated_tokens, metadata = self._fallback_generate(tokens)
           self.generation_stats['strategy_usage']['fallback'] += 1
       
       # Обновляем статистику
       generation_time = time.time() - start_time
       self._update_generation_stats(generated_tokens, generation_time, strategy)
       
       # Обновляем метаданные
       metadata.update({
           'generation_strategy': strategy,
           'generation_time': generation_time,
           'input_length': len(tokens),
           'output_length': len(generated_tokens),
           'tokens_added': len(generated_tokens) - len(tokens)
       })
       
       return ProcessingResult(
           data=generated_tokens,
           stage=ProcessingStage.GENERATION,
           modality=ModalityType.TEXT,
           metadata=metadata,
           component_name=self.name
       )
   
   def _select_generation_strategy(self, start_tokens: List[str]) -> str:
       """
       Выбирает стратегию генерации на основе контекста
       
       Args:
           start_tokens: Начальные токены
           
       Returns:
           Название стратегии
       """
       # Проверяем доступность данных для beam search
       has_sufficient_data = len(self.ngram_extractor.forward_graphs[2]) > 50
       
       # Длинные контексты лучше обрабатывать beam search
       context_bonus = len(start_tokens) > 3
       
       # Если недостаточно данных или короткий контекст, используем nucleus
       if not has_sufficient_data or not context_bonus:
           return self.gen_config.fallback_strategy
       
       return self.gen_config.generation_strategy
   
   def _beam_search_generate(self, start_tokens: List[str]) -> Tuple[List[str], Dict[str, Any]]:
       """
       Генерация с использованием beam search
       
       Args:
           start_tokens: Начальные токены
           
       Returns:
           Кортеж (сгенерированные_токены, метаданные)
       """
       start_time = time.time()
       
       # Инициализируем beam
       initial_candidate = BeamCandidate(
           tokens=start_tokens.copy(),
           score=0.0,
           normalized_score=0.0,
           metadata={'initial': True}
       )
       
       candidates = [initial_candidate]
       finished_candidates = []
       
       # Основной цикл beam search
       for step in range(self.gen_config.max_length - len(start_tokens)):
           if not candidates:
               break
           
           # Выполняем шаг beam search
           if self.gen_config.parallel_beam_search and len(candidates) > 1:
               candidates = self._parallel_beam_search_step(candidates)
               self.generation_stats['parallel_generations'] += 1
           else:
               candidates = self._beam_search_step(candidates)
           
           # Проверяем завершенные кандидаты
           continuing_candidates = []
           for candidate in candidates:
               if self._is_generation_complete(candidate):
                   finished_candidates.append(candidate)
               else:
                   continuing_candidates.append(candidate)
           
           candidates = continuing_candidates
           
           # Останавливаемся если есть достаточно завершенных кандидатов
           if len(finished_candidates) >= self.gen_config.beam_size:
               break
       
       # Добавляем оставшихся кандидатов к завершенным
       finished_candidates.extend(candidates)
       
       # Выбираем лучший результат
       if finished_candidates:
           best_candidate = max(finished_candidates, key=lambda x: x.normalized_score)
           result_tokens = best_candidate.tokens
       else:
           result_tokens = start_tokens
       
       # Метаданные
       generation_time = time.time() - start_time
       self.generation_stats['avg_beam_time'] = (
           (self.generation_stats['avg_beam_time'] * self.generation_stats['beam_search_usage'] + generation_time) /
           (self.generation_stats['beam_search_usage'] + 1)
       )
       
       metadata = {
           'method': 'beam_search',
           'beam_size': self.gen_config.beam_size,
           'candidates_explored': len(finished_candidates),
           'final_score': best_candidate.normalized_score if finished_candidates else 0,
           'parallel_processing': self.gen_config.parallel_beam_search,
           'steps_taken': step + 1 if 'step' in locals() else 0
       }
       
       return result_tokens, metadata
   
   def _beam_search_step(self, candidates: List[BeamCandidate]) -> List[BeamCandidate]:
       """
       Один шаг beam search
       
       Args:
           candidates: Текущие кандидаты
           
       Returns:
           Новые кандидаты после шага
       """
       new_candidates = []
       
       for candidate in candidates:
           # Получаем вероятности следующих токенов
           probabilities = self.ngram_extractor.get_interpolated_probabilities(candidate.tokens)
           
           if not probabilities:
               # Если нет вариантов, добавляем кандидата как есть
               new_candidates.append(candidate)
               continue
           
           # Создаем новых кандидатов для каждого возможного токена
           for token, prob in probabilities.items():
               if prob <= 0:
                   continue
               
               new_tokens = candidate.tokens + [token]
               
               # Вычисляем новый счет
               log_prob = math.log(prob + 1e-10)
               new_score = candidate.score + log_prob
               
               # Применяем length penalty
               length_penalty = self._calculate_length_penalty(new_tokens)
               normalized_score = new_score / length_penalty
               
               # Применяем repetition penalty
               if self.gen_config.repetition_penalty != 1.0:
                   repetition_factor = self._calculate_repetition_penalty(new_tokens, token)
                   normalized_score /= repetition_factor
               
               # Применяем diversity penalty
               if self.gen_config.diversity_penalty > 0:
                   diversity_factor = self._calculate_diversity_penalty(new_candidates, token)
                   normalized_score -= diversity_factor
               
               new_candidate = BeamCandidate(
                   tokens=new_tokens,
                   score=new_score,
                   normalized_score=normalized_score,
                   metadata={
                       'prob': prob,
                       'length_penalty': length_penalty,
                       'parent': candidate,
                       'step_added': len(new_tokens) - len(candidate.tokens)
                   }
               )
               
               new_candidates.append(new_candidate)
       
       # Возвращаем топ beam_size кандидатов
       new_candidates.sort(key=lambda x: x.normalized_score, reverse=True)
       return new_candidates[:self.gen_config.beam_size]
   
   def _parallel_beam_search_step(self, candidates: List[BeamCandidate]) -> List[BeamCandidate]:
       """
       Параллельный шаг beam search
       
       Args:
           candidates: Текущие кандидаты
           
       Returns:
           Новые кандидаты после параллельного шага
       """
       def process_candidate(candidate):
           local_candidates = []
           probabilities = self.ngram_extractor.get_interpolated_probabilities(candidate.tokens)
           
           for token, prob in probabilities.items():
               if prob <= 0:
                   continue
               
               new_tokens = candidate.tokens + [token]
               log_prob = math.log(prob + 1e-10)
               new_score = candidate.score + log_prob
               
               length_penalty = self._calculate_length_penalty(new_tokens)
               normalized_score = new_score / length_penalty
               
               if self.gen_config.repetition_penalty != 1.0:
                   repetition_factor = self._calculate_repetition_penalty(new_tokens, token)
                   normalized_score /= repetition_factor
               
               local_candidates.append(BeamCandidate(
                   tokens=new_tokens,
                   score=new_score,
                   normalized_score=normalized_score,
                   metadata={'prob': prob, 'length_penalty': length_penalty}
               ))
           
           return local_candidates
       
       # Параллельная обработка кандидатов
       all_new_candidates = []
       max_workers = min(self.gen_config.max_workers, len(candidates))
       
       with ThreadPoolExecutor(max_workers=max_workers) as executor:
           future_to_candidate = {
               executor.submit(process_candidate, candidate): candidate 
               for candidate in candidates
           }
           
           for future in as_completed(future_to_candidate):
               try:
                   local_candidates = future.result()
                   all_new_candidates.extend(local_candidates)
               except Exception as e:
                   logger.warning(f"Ошибка в параллельной обработке кандидата: {e}")
                   # Fallback на оригинального кандидата
                   original_candidate = future_to_candidate[future]
                   all_new_candidates.append(original_candidate)
       
       # Сортируем и возвращаем лучших
       all_new_candidates.sort(key=lambda x: x.normalized_score, reverse=True)
       return all_new_candidates[:self.gen_config.beam_size]
   
   def _nucleus_generate(self, start_tokens: List[str]) -> Tuple[List[str], Dict[str, Any]]:
       """
       Генерация с nucleus sampling
       
       Args:
           start_tokens: Начальные токены
           
       Returns:
           Кортеж (сгенерированные_токены, метаданные)
       """
       result = start_tokens.copy()
       steps_taken = 0
       
       for step in range(self.gen_config.max_length - len(start_tokens)):
           probabilities = self.ngram_extractor.get_interpolated_probabilities(result)
           
           if not probabilities:
               break
           
           next_token = self._nucleus_sampling(probabilities)
           if not next_token:
               break
           
           result.append(next_token)
           steps_taken += 1
           
           # Условие остановки
           if self._is_generation_complete_simple(result, next_token):
               break
       
       metadata = {
           'method': 'nucleus_sampling',
           'top_p': self.gen_config.top_p,
           'temperature': self.gen_config.temperature,
           'steps_taken': steps_taken
       }
       
       return result, metadata
   
   def _greedy_generate(self, start_tokens: List[str]) -> Tuple[List[str], Dict[str, Any]]:
       """
       Жадная генерация (выбор наиболее вероятного токена)
       
       Args:
           start_tokens: Начальные токены
           
       Returns:
           Кортеж (сгенерированные_токены, метаданные)
       """
       result = start_tokens.copy()
       steps_taken = 0
       
       for step in range(self.gen_config.max_length - len(start_tokens)):
           probabilities = self.ngram_extractor.get_interpolated_probabilities(result)
           
           if not probabilities:
               break
           
           # Выбираем токен с максимальной вероятностью
           next_token = max(probabilities.items(), key=lambda x: x[1])[0]
           result.append(next_token)
           steps_taken += 1
           
           if self._is_generation_complete_simple(result, next_token):
               break
       
       metadata = {
           'method': 'greedy',
           'steps_taken': steps_taken
       }
       
       return result, metadata
   
   def _fallback_generate(self, start_tokens: List[str]) -> Tuple[List[str], Dict[str, Any]]:
       """
       Fallback генерация при ошибках
       
       Args:
           start_tokens: Начальные токены
           
       Returns:
           Кортеж (сгенерированные_токены, метаданные)
       """
       # Простая генерация с минимальными требованиями
       result = start_tokens.copy()
       
       # Добавляем несколько общих токенов
       fallback_tokens = ["требует", "дальнейшего", "изучения"]
       max_add = min(len(fallback_tokens), self.gen_config.max_length - len(start_tokens))
       
       for i in range(max_add):
           result.append(fallback_tokens[i])
       
       metadata = {
           'method': 'fallback',
           'tokens_added': max_add,
           'fallback_reason': 'error_recovery'
       }
       
       return result, metadata
   
   def _nucleus_sampling(self, probabilities: Dict[str, float]) -> Optional[str]:
       """
       Nucleus (top-p) sampling для выбора следующего токена
       
       Args:
           probabilities: Словарь вероятностей токенов
           
       Returns:
           Выбранный токен или None
       """
       if not probabilities:
           return None
       
       # Сортируем по вероятности
       sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
       
       # Применяем temperature
       if self.gen_config.temperature != 1.0:
           adjusted_probs = {}
           for token, prob in sorted_items:
               adjusted_probs[token] = prob ** (1.0 / self.gen_config.temperature)
           
           # Нормализуем
           total_prob = sum(adjusted_probs.values())
           if total_prob > 0:
               for token in adjusted_probs:
                   adjusted_probs[token] /= total_prob
           
           sorted_items = sorted(adjusted_probs.items(), key=lambda x: x[1], reverse=True)
       
       # Nucleus sampling
       cumulative_prob = 0
       selected_tokens = []
       
       for token, prob in sorted_items:
           cumulative_prob += prob
           selected_tokens.append((token, prob))
           
           if cumulative_prob >= self.gen_config.top_p:
               break
       
       # Обеспечиваем минимальное количество токенов
       if len(selected_tokens) < self.gen_config.min_tokens_to_keep:
           selected_tokens = sorted_items[:self.gen_config.min_tokens_to_keep]
       
       # Выбираем случайный токен из отобранных
       if selected_tokens:
           tokens, probs = zip(*selected_tokens)
           return random.choices(tokens, weights=probs)[0]
       
       return None
   
   def _calculate_length_penalty(self, tokens: List[str]) -> float:
       """Вычисляет штраф за длину"""
       return ((len(tokens) + 5) / 6) ** self.gen_config.length_penalty
   
   def _calculate_repetition_penalty(self, tokens: List[str], new_token: str) -> float:
       """Вычисляет штраф за повторения"""
       if self.gen_config.repetition_penalty == 1.0:
           return 1.0
       
       # Проверяем последние несколько токенов
       recent_tokens = tokens[-10:]  # Смотрим на последние 10 токенов
       repetition_count = recent_tokens.count(new_token)
       
       if repetition_count > 0:
           return self.gen_config.repetition_penalty ** repetition_count
       
       return 1.0
   
   def _calculate_diversity_penalty(self, candidates: List[BeamCandidate], token: str) -> float:
       """Вычисляет штраф за недостаток разнообразия"""
       if self.gen_config.diversity_penalty == 0:
           return 0.0
       
       # Считаем сколько кандидатов уже используют этот токен
       token_usage = sum(1 for candidate in candidates 
                        if candidate.tokens and candidate.tokens[-1] == token)
       
       return self.gen_config.diversity_penalty * token_usage
   
   def _is_generation_complete(self, candidate: BeamCandidate) -> bool:
       """Проверяет завершенность генерации для beam candidate"""
       tokens = candidate.tokens
       
       # Проверяем максимальную длину
       if len(tokens) >= self.gen_config.max_length:
           return True
       
       # Проверяем раннюю остановку
       if (self.gen_config.early_stopping and 
           len(tokens) >= self.gen_config.min_length and 
           tokens[-1] in ['.', '!', '?']):
           return True
       
       return False
   
   def _is_generation_complete_simple(self, tokens: List[str], last_token: str) -> bool:
       """Простая проверка завершенности генерации"""
       return (len(tokens) >= self.gen_config.min_length and 
               last_token in ['.', '!', '?'] and 
               self.gen_config.early_stopping)
   
   def _update_generation_stats(self, generated_tokens: List[str], 
                               generation_time: float, strategy: str) -> None:
       """Обновляет статистику генерации"""
       self.generation_stats['total_generations'] += 1
       
       # Обновляем среднее время генерации
       total_gens = self.generation_stats['total_generations']
       old_avg = self.generation_stats['avg_generation_time']
       self.generation_stats['avg_generation_time'] = (
           (old_avg * (total_gens - 1) + generation_time) / total_gens
       )
       
       # Статистика длины
       gen_length = len(generated_tokens)
       length_stats = self.generation_stats['length_stats']
       
       old_avg_length = length_stats['avg_generated_length']
       length_stats['avg_generated_length'] = (
           (old_avg_length * (total_gens - 1) + gen_length) / total_gens
       )
       length_stats['min_generated'] = min(length_stats['min_generated'], gen_length)
       length_stats['max_generated'] = max(length_stats['max_generated'], gen_length)
   
   def get_generation_stats(self) -> Dict[str, Any]:
       """Возвращает статистику генерации"""
       stats = super().get_stats()
       stats.update({
           'generation_stats': self.generation_stats.copy(),
           'config': {
               'beam_size': self.gen_config.beam_size,
               'top_p': self.gen_config.top_p,
               'temperature': self.gen_config.temperature,
               'max_length': self.gen_config.max_length,
               'strategy': self.gen_config.generation_strategy
           }
       })
       return stats
   
   def update_config(self, new_config: Dict[str, Any]) -> None:
       """
       Обновляет конфигурацию генератора
       
       Args:
           new_config: Новые параметры конфигурации
       """
       old_config = self.gen_config
       
       try:
           # Обновляем конфигурацию
           gen_updates = new_config.get('generation', {})
           
           for key, value in gen_updates.items():
               if hasattr(self.gen_config, key):
                   setattr(self.gen_config, key, value)
           
           logger.info(f"Конфигурация генератора {self.name} обновлена: {gen_updates}")
           
       except Exception as e:
           # Откатываем изменения при ошибке
           self.gen_config = old_config
           raise ValueError(f"Ошибка обновления конфигурации: {e}")
   
   def _get_default_stage(self) -> ProcessingStage:
       """Генерация - это этап генерации"""
       return ProcessingStage.GENERATION