# ARIA - Adaptive Reconfigurable Intelligence Architecture

## 🌟 Обзор

**ARIA** (Adaptive Reconfigurable Intelligence Architecture) — это современная графовая архитектура для мультимодальной обработки данных с декларативным DSL и модульным дизайном. Система позволяет создавать сложные пайплайны обработки данных, объединяя текст, изображения, аудио и другие модальности в единой экосистеме.

## ⭐ Ключевые особенности

- 🧩 **Модульная архитектура** - независимые переиспользуемые компоненты
- 🌐 **Графовая обработка** - DAG с автоматической оптимизацией выполнения
- 🎨 **Мультимодальность** - единый подход для текста, изображений, аудио, видео
- 📝 **Декларативный DSL** - YAML/JSON конфигурации + Fluent API
- 🚀 **Высокая производительность** - параллельная обработка и кэширование
- 🔧 **Расширяемость** - простое добавление новых компонентов и модальностей
- 📊 **Мониторинг** - детальная статистика и отладочная информация

## 🏗️ Архитектура

### Основные компоненты

```
ARIA Architecture
├── Core System
│   ├── ProcessingGraph - DAG для выполнения пайплайнов
│   ├── ComponentRegistry - реестр всех компонентов
│   └── GraphFactory - создание графов из конфигураций
├── Modality Processors
│   ├── Text Processing (N-grams, Tokenization, Generation)
│   ├── Image Processing (Patches, Features, Encoding)
│   ├── Audio Processing (STFT, MFCC, Spectral)
│   └── Video Processing (Frame Extraction, Temporal)
├── Fusion Strategies
│   ├── Cross-Modal Attention
│   ├── Modality-Specific Fusion
│   └── Generic Concatenation
├── DSL System
│   ├── YAML/JSON Parser
│   ├── Fluent API Builder
│   ├── Template Manager
│   └── CLI Interface
└── Extensions
    ├── Performance Monitoring
    ├── Error Handling
    └── Configuration Management
```

### Типы данных

- **ModalityData** - универсальный контейнер для данных любой модальности
- **ProcessingResult** - результат обработки с метаданными
- **Component** - базовый класс для всех обработчиков
- **ProcessingGraph** - граф выполнения с топологической сортировкой

## 🚀 Быстрый старт

### Установка

```bash
git clone https://github.com/your-org/aria
cd aria
pip install -r requirements.txt
```

### Простой пример

```python
from aria.core import GLOBAL_REGISTRY
from aria.text_components import register_text_components
from aria.dsl_system import FluentPipelineBuilder

# Регистрируем компоненты
register_text_components()

# Создаем пайплайн
pipeline = (FluentPipelineBuilder(GLOBAL_REGISTRY, "my_pipeline")
            .tokenizer("tokenizer", preserve_punctuation=True)
            .ngram_extractor("ngrams", ngram={'min_n': 2, 'max_n': 5})
            .generator("generator", generation={'beam_size': 5})
            .build())

# Выполняем
input_data = {"text_input": "машинное обучение это"}
results = pipeline.execute(input_data)
```

## 📝 DSL и конфигурация

### YAML конфигурация

```yaml
variables:
  beam_size: 5
  max_length: 20

pipeline:
  name: "text_generation_pipeline"
  
  components:
    - name: "tokenizer"
      type: "AdvancedTextTokenizer"
      params:
        preserve_punctuation: true
        min_token_length: 2
      inputs: ["text_input"]
      outputs: ["tokens"]
    
    - name: "ngram_processor"
      type: "NGramExtractor"
      params:
        ngram:
          min_n: 2
          max_n: 5
          dynamic_selection: true
      inputs: ["tokens"]
      outputs: ["ngram_data"]
    
    - name: "generator"
      type: "BeamSearchGenerator"
      params:
        generation:
          beam_size: ${beam_size}
          max_length: ${max_length}
      inputs: ["tokens"]
      outputs: ["generated_text"]
  
  connections:
    - ["tokenizer", "ngram_processor"]
    - ["tokenizer", "generator"]
```

### Использование YAML

```python
from aria.dsl_system import PipelineDSL

dsl = PipelineDSL(GLOBAL_REGISTRY)
pipeline = dsl.parse_file("config.yaml")
results = pipeline.execute({"text_input": "привет мир"})
```

### Fluent API

```python
pipeline = (FluentPipelineBuilder(registry, "fluent_example")
    .tokenizer("tok1", preserve_punctuation=True)
    .ngram_extractor("ngram1", ngram={'min_n': 2, 'max_n': 4})
    .generator("gen1", generation={'beam_size': 3})
    .fusion("fusion1", method="concatenation")
    .build())
```

## 🌈 Мультимодальная обработка

### Пример: текст + изображение

```python
from aria.multimodal_extensions import register_multimodal_components, create_multimodal_pipeline

# Регистрируем мультимодальные компоненты
register_multimodal_components()

# Создаем пайплайн из шаблона
pipeline = create_multimodal_pipeline("image_text_fusion")

# Обрабатываем мультимодальные данные
input_data = {
    "image_input": np.array([...]),  # RGB изображение
    "text_input": "описание изображения"
}

results = pipeline.execute(input_data)
```

### Кросс-модальное внимание

```yaml
components:
  - name: "image_encoder"
    type: "ImageEncoder"
    params:
      feature_type: "histogram"
    inputs: ["image_patches"]
    outputs: ["image_features"]
  
  - name: "text_encoder"
    type: "NGramExtractor"
    inputs: ["text_tokens"]
    outputs: ["text_features"]
  
  - name: "cross_attention"
    type: "CrossModalAttention"
    params:
      attention_dim: 128
      temperature: 1.0
    inputs: ["image_features", "text_features"]
    outputs: ["fused_features"]
```

## 🧩 Компоненты

### Текстовые компоненты

- **AdvancedTextTokenizer** - продвинутая токенизация с сохранением пунктуации
- **NGramExtractor** - извлечение n-грамм с адаптивными весами
- **BeamSearchGenerator** - генерация с beam search и nucleus sampling
- **TextFusionStrategy** - стратегии слияния текстовых данных

### Мультимодальные компоненты

- **ImageTokenizer** - разбиение изображений на патчи
- **ImageEncoder** - извлечение визуальных признаков
- **AudioTokenizer** - спектральный анализ аудио
- **AudioEncoder** - MFCC и спектральные признаки
- **VideoTokenizer** - извлечение кадров из видео

### Стратегии слияния

- **CrossModalAttention** - кросс-модальное внимание
- **ModalitySpecificFusion** - специализированное слияние по типам модальностей
- **MultimodalGenerator** - генерация на основе мультимодального контекста

## 🔧 Создание собственных компонентов

### Базовый компонент

```python
from aria.core import Component, ProcessingResult, ProcessingStage

class MyCustomComponent(Component):
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.my_param = config.get('my_param', 'default_value')
    
    def process(self, input_data: Any) -> ProcessingResult:
        # Ваша логика обработки
        processed_data = self._my_processing_logic(input_data)
        
        return ProcessingResult(
            data=processed_data,
            stage=ProcessingStage.ENCODING,
            modality=ModalityType.TEXT,
            metadata={"custom_info": "example"}
        )
    
    def _my_processing_logic(self, data):
        # Реализация обработки
        return data

# Регистрация компонента
GLOBAL_REGISTRY.register("MyCustomComponent", MyCustomComponent)
```

### Мультимодальный компонент

```python
from aria.core import MultiModalComponent, ModalityType

class MyMultimodalComponent(MultiModalComponent):
    def __init__(self, name: str, config: Dict[str, Any] = None):
        supported_modalities = [ModalityType.TEXT, ModalityType.IMAGE]
        super().__init__(name, supported_modalities, config)
    
    def process(self, input_data: ModalityData) -> ProcessingResult:
        if not self.validate_input(input_data):
            raise ValueError(f"Неподдерживаемая модальность: {input_data.modality}")
        
        # Обработка в зависимости от модальности
        if input_data.modality == ModalityType.TEXT:
            result = self._process_text(input_data.data)
        elif input_data.modality == ModalityType.IMAGE:
            result = self._process_image(input_data.data)
        
        return ProcessingResult(
            data=result,
            stage=ProcessingStage.ENCODING,
            modality=input_data.modality
        )
```

## 📊 CLI интерфейс

ARIA предоставляет мощный CLI для управления пайплайнами:

```bash
# Создание пайплайна из шаблона
python -m aria create simple_text_generation

# Загрузка из файла конфигурации
python -m aria load my_pipeline.yaml

# Запуск пайплайна
python -m aria run "текст для обработки"

# Информация о пайплайне
python -m aria info

# Список доступных шаблонов
python -m aria templates

# Список компонентов
python -m aria components
```

### Интерактивный режим

```bash
python -m aria interactive
```

```
> create simple_text_generation
✅ Pipeline created from template 'simple_text_generation'

> run машинное обучение
🔄 Processing...
✅ tokenizer: Success (0.003s)
✅ ngram_extractor: Success (0.015s)
✅ generator: Success (0.089s)

> info
Pipeline: simple_text_generation
Nodes: 3
Connections: 2
Execution order: tokenizer -> ngram_extractor -> generator
```

## 🎯 Примеры использования

### Генерация текста

```python
# Простая генерация
pipeline = create_text_generation_pipeline()
result = pipeline.execute({"text_input": "искусственный интеллект"})

# Настройка параметров
config = {
    "generation": {
        "beam_size": 8,
        "temperature": 0.7,
        "max_length": 25
    }
}
pipeline = create_text_generation_pipeline(config)
```

### Анализ изображений

```python
# Извлечение признаков
pipeline = create_image_analysis_pipeline()
result = pipeline.execute({
    "image_input": load_image("photo.jpg")
})

# Мультимодальный анализ
pipeline = create_multimodal_pipeline("image_text_fusion")
result = pipeline.execute({
    "image_input": load_image("photo.jpg"),
    "text_input": "что изображено на фото"
})
```

### Обработка аудио

```python
# Анализ аудио
pipeline = create_audio_pipeline()
result = pipeline.execute({
    "audio_input": "speech.wav"
})

# Аудио + текст
pipeline = create_multimodal_pipeline("audio_text_analysis")
result = pipeline.execute({
    "audio_input": "speech.wav",
    "text_input": "транскрипция речи"
})
```

## 📈 Производительность и мониторинг

### Статистика выполнения

```python
# Получение детальной статистики
results = pipeline.execute(input_data)
for component, stats in results.items():
    if stats["status"] == "success":
        print(f"{component}: {stats['processing_time']:.3f}s")
```

### Профилирование

```python
# Бенчмарк производительности
from aria.utils import benchmark_pipeline

benchmark_results = benchmark_pipeline(pipeline, test_inputs)
print(f"Среднее время: {benchmark_results['avg_time']:.3f}s")
print(f"Пропускная способность: {benchmark_results['throughput']:.1f} items/sec")
```

## 🛠️ Расширенные возможности

### Шаблоны

ARIA включает готовые шаблоны для типичных задач:

- `simple_text_generation` - базовая генерация текста
- `model_training` - обучение модели
- `multimodal_fusion` - слияние модальностей
- `image_text_fusion` - анализ изображений с текстом
- `audio_text_analysis` - обработка аудио и текста
- `trimodal_processing` - обработка трех модальностей

### Настройка шаблонов

```python
# Переопределение параметров шаблона
overrides = {
    "components.1.params.ngram.max_n": 6,
    "components.2.params.generation.beam_size": 10
}

pipeline = template_manager.create_from_template(
    "simple_text_generation", 
    registry, 
    overrides
)
```

### Обработка ошибок

```python
# Graceful degradation при ошибках
try:
    results = pipeline.execute(input_data)
    successful = [name for name, result in results.items() 
                 if result["status"] == "success"]
    print(f"Успешно: {len(successful)}/{len(results)} компонентов")
except Exception as e:
    print(f"Критическая ошибка: {e}")
```

## 🧪 Тестирование

### Unit тесты

```bash
python -m pytest tests/test_core.py
python -m pytest tests/test_text_components.py
python -m pytest tests/test_multimodal.py
```

### Интеграционные тесты

```bash
python -m pytest tests/test_pipelines.py
python -m pytest tests/test_dsl.py
```

### Тестирование производительности

```bash
python -m aria benchmark
```

## 📚 Документация

### API Reference

- [Core Components](docs/core.md)
- [Text Processing](docs/text.md)
- [Multimodal Extensions](docs/multimodal.md)
- [DSL System](docs/dsl.md)

### Tutorials

- [Getting Started](docs/tutorials/getting_started.md)
- [Creating Custom Components](docs/tutorials/custom_components.md)
- [Multimodal Processing](docs/tutorials/multimodal.md)
- [Advanced Configuration](docs/tutorials/advanced_config.md)

## 🤝 Участие в разработке

### Добавление нового компонента

1. Создайте класс, наследующий от `Component` или `MultiModalComponent`
2. Реализуйте метод `process()`
3. Добавьте регистрацию в соответствующий модуль
4. Напишите тесты
5. Обновите документацию

### Добавление новой модальности

1. Добавьте новый `ModalityType` в `core.py`
2. Создайте токенизатор и энкодер для модальности
3. Добавьте стратегии слияния
4. Создайте шаблоны пайплайнов
5. Добавьте примеры использования

## 🐛 Решение проблем

### Частые проблемы

**Q: Компонент не найден**
```
ValueError: Неизвестный тип компонента: MyComponent
```
A: Убедитесь, что компонент зарегистрирован в `GLOBAL_REGISTRY`

**Q: Циклическая зависимость**
```
ValueError: Циклическая зависимость обнаружена в узле: component1
```
A: Проверьте connections в конфигурации на циклы

**Q: Ошибка размерности при слиянии**
```
ValueError: Cannot concatenate arrays with different shapes
```
A: Используйте нормализацию размерностей в стратегии слияния

### Отладка

```python
# Включение отладочной информации
import logging
logging.basicConfig(level=logging.DEBUG)

# Проверка конфигурации графа
graph_info = pipeline.get_graph_info()
print(f"Execution order: {graph_info['execution_order']}")

# Анализ ошибок выполнения
results = pipeline.execute(input_data)
for component, result in results.items():
    if result["status"] == "error":
        print(f"Error in {component}: {result['error']}")
```

## 📄 Лицензия

MIT License - см. [LICENSE](LICENSE)

## 📞 Контакты

- **Основатель**: [Ваше имя]
- **Email**: your.email@example.com
- **GitHub**: https://github.com/your-org/aria
- **Документация**: https://aria-docs.example.com

## 🙏 Благодарности

- Команда разработчиков ARIA
- Сообщество пользователей
- Контрибьюторы open-source проектов

---

**ARIA** - делаем мультимодальную обработку данных простой, гибкой и мощной! 🚀