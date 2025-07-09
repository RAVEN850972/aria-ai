# aria/processors/text/tokenizers.py
"""
Текстовые токенизаторы для ARIA
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from collections import Counter

from ...core import MultiModalComponent, ModalityType, ModalityData, ProcessingResult, ProcessingStage

logger = logging.getLogger(__name__)


class AdvancedTextTokenizer(MultiModalComponent):
    """
    Продвинутый текстовый токенизатор с сохранением контекста
    
    Возможности:
    - Сохранение знаков препинания как отдельных токенов
    - Нормализация пробелов и регистра
    - Фильтрация по минимальной длине токенов
    - Обработка специальных символов
    - Статистика токенизации
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Инициализация токенизатора
        
        Args:
            name: Имя компонента
            config: Конфигурация токенизатора
        """
        super().__init__(name, [ModalityType.TEXT], config)
        
        # Параметры токенизации
        self.preserve_punctuation = config.get('preserve_punctuation', True)
        self.min_token_length = config.get('min_token_length', 2)
        self.normalize_case = config.get('normalize_case', True)
        self.remove_extra_spaces = config.get('remove_extra_spaces', True)
        self.split_contractions = config.get('split_contractions', False)
        self.handle_urls = config.get('handle_urls', True)
        self.handle_emails = config.get('handle_emails', True)
        self.custom_separators = config.get('custom_separators', [])
        
        # Специальные токены
        self.special_tokens = {
            '<UNK>': 'неизвестный токен',
            '<NUM>': 'число',
            '<URL>': 'веб-адрес',
            '<EMAIL>': 'электронная почта',
            '<PUNCT>': 'знак препинания'
        }
        
        # Статистика
        self.stats = {
            'total_texts': 0,
            'total_tokens': 0,
            'unique_tokens': set(),
            'avg_tokens_per_text': 0.0,
            'punctuation_tokens': 0,
            'special_tokens_used': Counter()
        }
        
        logger.debug(f"Инициализирован токенизатор {name} с конфигурацией: {config}")
    
    def process(self, input_data: ModalityData) -> ProcessingResult:
        """
        Токенизация текстовых данных
        
        Args:
            input_data: Данные для токенизации
            
        Returns:
            ProcessingResult с токенами
        """
        if not self.validate_input(input_data):
            raise ValueError(f"Неподдерживаемая модальность: {input_data.modality}")
        
        text = input_data.data
        if not isinstance(text, str):
            raise ValueError("Данные должны быть строкой")
        
        # Выполняем токенизацию
        tokens = self._tokenize_advanced(text)
        
        # Обновляем статистику
        self._update_stats(text, tokens)
        
        # Создаем результат
        return ProcessingResult(
            data=tokens,
            stage=ProcessingStage.TOKENIZATION,
            modality=ModalityType.TEXT,
            metadata={
                "token_count": len(tokens),
                "original_length": len(text),
                "unique_tokens": len(set(tokens)),
                "punctuation_ratio": self._calculate_punctuation_ratio(tokens),
                "tokenizer_config": {
                    "preserve_punctuation": self.preserve_punctuation,
                    "min_token_length": self.min_token_length,
                    "normalize_case": self.normalize_case
                }
            },
            component_name=self.name
        )
    
    def _tokenize_advanced(self, text: str) -> List[str]:
        """
        Выполняет продвинутую токенизацию
        
        Args:
            text: Исходный текст
            
        Returns:
            Список токенов
        """
        # Сохраняем исходный текст для отладки
        original_text = text
        
        # Предобработка специальных случаев
        text = self._preprocess_special_cases(text)
        
        # Обработка знаков препинания
        if self.preserve_punctuation:
            text = self._handle_punctuation(text)
        
        # Обработка сокращений
        if self.split_contractions:
            text = self._split_contractions(text)
        
        # Нормализация пробелов
        if self.remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text.strip())
        
        # Нормализация регистра
        if self.normalize_case:
            text = text.lower()
        
        # Разбиение на токены
        tokens = text.split()
        
        # Дополнительные разделители
        if self.custom_separators:
            tokens = self._apply_custom_separators(tokens)
        
        # Фильтрация токенов
        filtered_tokens = self._filter_tokens(tokens)
        
        logger.debug(f"Токенизация: '{original_text[:50]}...' -> {len(filtered_tokens)} токенов")
        
        return filtered_tokens
    
    def _preprocess_special_cases(self, text: str) -> str:
        """Предобработка специальных случаев"""
        # Обработка URL
        if self.handle_urls:
            url_pattern = r'https?://[^\s]+'
            text = re.sub(url_pattern, ' <URL> ', text)
        
        # Обработка email
        if self.handle_emails:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            text = re.sub(email_pattern, ' <EMAIL> ', text)
        
        # Обработка чисел
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        text = re.sub(number_pattern, ' <NUM> ', text)
        
        return text
    
    def _handle_punctuation(self, text: str) -> str:
        """Обработка знаков препинания"""
        # Основные знаки препинания
        punctuation_marks = ['.', '!', '?', ',', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'"]
        
        for mark in punctuation_marks:
            # Добавляем пробелы вокруг знаков препинания
            text = text.replace(mark, f' {mark} ')
        
        # Специальная обработка многоточий
        text = re.sub(r'\.{3,}', ' ... ', text)
        
        # Обработка кавычек
        text = re.sub(r'[«»„‚"]', ' " ', text)
        
        return text
    
    def _split_contractions(self, text: str) -> str:
        """Разделение сокращений"""
        contractions = {
            "can't": "cannot",
            "won't": "will not", 
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _apply_custom_separators(self, tokens: List[str]) -> List[str]:
        """Применяет дополнительные разделители"""
        result = []
        
        for token in tokens:
            # Разбиваем по дополнительным разделителям
            parts = [token]
            for separator in self.custom_separators:
                new_parts = []
                for part in parts:
                    new_parts.extend(part.split(separator))
                parts = [p for p in new_parts if p]  # Убираем пустые
            
            result.extend(parts)
        
        return result
    
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """Фильтрует токены по различным критериям"""
        filtered = []
        
        for token in tokens:
            # Пропускаем пустые токены
            if not token.strip():
                continue
            
            # Проверяем минимальную длину (кроме специальных токенов и знаков препинания)
            if (len(token) < self.min_token_length and 
                token not in self.special_tokens and
                not self._is_punctuation(token)):
                continue
            
            filtered.append(token)
        
        return filtered
    
    def _is_punctuation(self, token: str) -> bool:
        """Проверяет, является ли токен знаком препинания"""
        punctuation_chars = set('.,!?;:()[]{}"\'-–—…«»""„"‚')
        return len(token) <= 3 and all(c in punctuation_chars for c in token)
    
    def _calculate_punctuation_ratio(self, tokens: List[str]) -> float:
        """Вычисляет долю знаков препинания среди токенов"""
        if not tokens:
            return 0.0
        
        punctuation_count = sum(1 for token in tokens if self._is_punctuation(token))
        return punctuation_count / len(tokens)
    
    def _update_stats(self, text: str, tokens: List[str]) -> None:
        """Обновляет статистику токенизации"""
        self.stats['total_texts'] += 1
        self.stats['total_tokens'] += len(tokens)
        self.stats['unique_tokens'].update(tokens)
        
        # Считаем знаки препинания
        punctuation_count = sum(1 for token in tokens if self._is_punctuation(token))
        self.stats['punctuation_tokens'] += punctuation_count
        
        # Считаем специальные токены
        for token in tokens:
            if token in self.special_tokens:
                self.stats['special_tokens_used'][token] += 1
        
        # Обновляем среднее количество токенов
        self.stats['avg_tokens_per_text'] = self.stats['total_tokens'] / self.stats['total_texts']
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Восстанавливает текст из токенов
        
        Args:
            tokens: Список токенов
            
        Returns:
            Восстановленный текст
        """
        if not tokens:
            return ""
        
        result = []
        
        for i, token in enumerate(tokens):
            # Специальная обработка знаков препинания
            if self._is_punctuation(token):
                # Знаки препинания обычно не отделяются пробелом слева
                if result and not self._needs_space_before(token):
                    result[-1] += token
                else:
                    result.append(token)
            else:
                result.append(token)
        
        text = ' '.join(result)
        
        # Постобработка для улучшения читаемости
        text = self._postprocess_detokenized_text(text)
        
        return text
    
    def _needs_space_before(self, token: str) -> bool:
        """Определяет, нужен ли пробел перед токеном"""
        # Открывающие знаки обычно нуждаются в пробеле перед собой
        opening_marks = {'(', '[', '{', '"', '«', '"', '„'}
        return token in opening_marks
    
    def _postprocess_detokenized_text(self, text: str) -> str:
        """Постобработка восстановленного текста"""
        # Убираем лишние пробелы перед знаками препинания
        text = re.sub(r'\s+([.!?,:;)])', r'\1', text)
        
        # Убираем лишние пробелы после открывающих скобок
        text = re.sub(r'([([])\s+', r'\1', text)
        
        # Нормализуем множественные пробелы
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def get_vocabulary(self, min_frequency: int = 1) -> Dict[str, int]:
        """
        Возвращает словарь токенов с частотами
        
        Args:
            min_frequency: Минимальная частота токена
            
        Returns:
            Словарь {токен: частота}
        """
        vocab = {}
        
        # Подсчитываем частоты (упрощенная реализация)
        # В реальной реализации можно хранить более детальную статистику
        for token in self.stats['unique_tokens']:
            # Здесь должна быть реальная частота, пока используем заглушку
            freq = 1  # Заглушка
            if freq >= min_frequency:
                vocab[token] = freq
        
        return vocab
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику токенизатора"""
        stats = super().get_stats()
        stats.update({
            'tokenization_stats': {
                'total_texts_processed': self.stats['total_texts'],
                'total_tokens_generated': self.stats['total_tokens'],
                'unique_tokens_count': len(self.stats['unique_tokens']),
                'avg_tokens_per_text': self.stats['avg_tokens_per_text'],
                'punctuation_tokens': self.stats['punctuation_tokens'],
                'special_tokens_usage': dict(self.stats['special_tokens_used']),
                'vocabulary_size': len(self.stats['unique_tokens'])
            }
        })
        return stats
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику токенизатора"""
        super().reset_stats()
        self.stats = {
            'total_texts': 0,
            'total_tokens': 0,
            'unique_tokens': set(),
            'avg_tokens_per_text': 0.0,
            'punctuation_tokens': 0,
            'special_tokens_used': Counter()
        }
    
    def _get_default_stage(self) -> ProcessingStage:
        """Токенизация - это этап токенизации"""
        return ProcessingStage.TOKENIZATION