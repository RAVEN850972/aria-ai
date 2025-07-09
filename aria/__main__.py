# aria/__main__.py
"""Entry point для ARIA CLI"""

import sys
import argparse

from .core import GLOBAL_REGISTRY
from .processors import register_all_processors
from .dsl import PipelineCLI, FluentPipelineBuilder, create_text_pipeline


def main():
    """Главная функция CLI"""
    parser = argparse.ArgumentParser(description='ARIA - Adaptive Reconfigurable Intelligence Architecture')
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Команда create
    create_parser = subparsers.add_parser('create', help='Создать пайплайн из шаблона')
    create_parser.add_argument('template', help='Имя шаблона')
    
    # Команда run
    run_parser = subparsers.add_parser('run', help='Запустить обработку текста')
    run_parser.add_argument('text', nargs='+', help='Текст для обработки')
    
    # Команда load
    load_parser = subparsers.add_parser('load', help='Загрузить пайплайн из файла')
    load_parser.add_argument('file', help='Путь к файлу конфигурации')
    
    # Команды без аргументов
    subparsers.add_parser('info', help='Информация о пайплайне')
    subparsers.add_parser('templates', help='Список шаблонов')
    subparsers.add_parser('components', help='Список компонентов')
    subparsers.add_parser('interactive', help='Интерактивный режим')
    subparsers.add_parser('demo', help='Демо режим')
    
    # Парсим аргументы
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Инициализация системы
    register_all_processors()
    cli = PipelineCLI(GLOBAL_REGISTRY)
    
    # Выполняем команды
    if args.command == 'create':
        result = cli.execute_command('create', [args.template])
        print(result)
    
    elif args.command == 'run':
        result = cli.execute_command('run', args.text)
        print(result)
    
    elif args.command == 'load':
        result = cli.execute_command('load', [args.file])
        print(result)
    
    elif args.command in ['info', 'templates', 'components']:
        result = cli.execute_command(args.command, [])
        print(result)
    
    elif args.command == 'interactive':
        interactive_mode(cli)
    
    elif args.command == 'demo':
        demo_mode()


def interactive_mode(cli: PipelineCLI):
    """Интерактивный режим"""
    print("🚀 ARIA Interactive Mode")
    print("Команды: create, run, load, info, templates, components, exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("aria> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ('exit', 'quit'):
                print("👋 До свидания!")
                break
            
            # Парсим команду
            parts = user_input.split()
            command = parts[0]
            args = parts[1:]
            
            result = cli.execute_command(command, args)
            print(result)
            
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


def demo_mode():
    """Демонстрационный режим"""
    print("🎯 ARIA Demo Mode")
    print("=" * 40)
    
    # Регистрируем компоненты
    register_all_processors()
    
    # Создаем простой пайплайн с правильными входами
    print("📦 Создание пайплайна...")
    pipeline = (FluentPipelineBuilder(GLOBAL_REGISTRY, "demo_pipeline")
                .tokenizer("demo_tokenizer", preserve_punctuation=True)
                .ngram_extractor("demo_ngrams", ngram={'min_n': 2, 'max_n': 4})
                .generator("demo_generator", generation={'beam_size': 3})
                .build())
    
    print("✅ Пайплайн создан!")
    
    # Тестовые фразы
    test_phrases = [
        "машинное обучение",
        "искусственный интеллект", 
        "обработка естественного языка",
        "нейронные сети"
    ]
    
    print("\n🧪 Тестирование генерации:")
    print("-" * 30)
    
    from .core import ModalityData, ModalityType
    
    for phrase in test_phrases:
        print(f"\n📝 Вход: '{phrase}'")
        
        try:
            # Создаем правильный входной формат
            input_data = {
                "input": ModalityData(
                    data=phrase,
                    modality=ModalityType.TEXT
                )
            }
            
            results = pipeline.execute(input_data)
            
            # Показываем успешные результаты
            success_count = 0
            for node_name, result in results.items():
                if result["status"] == "success":
                    print(f"   ✅ {node_name}: OK")
                    success_count += 1
                else:
                    print(f"   ❌ {node_name}: {result.get('error', 'Unknown error')}")
            
            print(f"   🎯 Результат: {success_count}/{len(results)} успешно")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    print(f"\n🎉 Демо завершено!")


def quick_start():
    """Быстрый старт для разработчиков"""
    register_all_processors()
    
    # Простой пример использования
    pipeline = create_text_pipeline(GLOBAL_REGISTRY)
    result = pipeline.execute({"text_input": "пример текста"})
    
    return pipeline, result


if __name__ == '__main__':
    main()