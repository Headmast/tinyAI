#!/usr/bin/env python3
"""
Тестовый скрипт для проверки GPT-5.4 с 4 температурами (0, 0.5, 0.75, 1.0)
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from llm_cli import (
    compare_temperatures,
    setup_logging,
    get_available_models,
    get_client_for_model
)

def main():
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ GPT-5.4 С 4 ТЕМПЕРАТУРАМИ")
    print("=" * 70)
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("\n❌ ОШИБКА: OPENAI_API_KEY не найден в .env")
        print("Для тестирования GPT-5.4 необходим API ключ OpenAI")
        return 1
    
    print("\n✓ API ключ OpenAI найден")
    
    clients = {
        "openai": OpenAI(api_key=openai_api_key, timeout=60.0)
    }
    
    model_name = "gpt-5.4"
    models = get_available_models()
    
    if model_name not in models:
        print(f"\n❌ ОШИБКА: Модель {model_name} не найдена в списке")
        return 1
    
    print(f"✓ Используемая модель: {models[model_name]['name']}")
    print(f"✓ Провайдер: {models[model_name]['provider']}")
    print(f"✓ Цена: ${models[model_name]['prompt_price']}/1K prompt, ${models[model_name]['completion_price']}/1K completion")
    
    log_file = setup_logging()
    print(f"✓ Лог-файл создан: {log_file}")
    
    test_question = "Как лучше организовывать свой рабочий день, бороться с прокрастинацией и быть успешным?"
    
    print(f"\n✓ Тестовый вопрос: {test_question}")
    print(f"✓ Температуры для тестирования: 0, 0.5, 0.75, 1.0")
    print("\n" + "=" * 70)
    print("НАЧАЛО ТЕСТИРОВАНИЯ")
    print("=" * 70)
    
    try:
        client = get_client_for_model(model_name, clients)
        results = compare_temperatures(client, test_question, log_file, model_name)
        
        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        print("=" * 70)
        
        if results:
            print(f"\n✓ Успешно выполнено {len(results)} запросов")
            print(f"✓ Все результаты записаны в: {log_file}")
            
            print("\n📊 Краткая сводка:")
            total_cost = 0
            for result in results:
                cost = (21 * models[model_name]['prompt_price'] / 1000) + \
                       (result['tokens'] * models[model_name]['completion_price'] / 1000)
                total_cost += cost
                print(f"\n  Temperature {result['temperature']}:")
                print(f"    - Символов: {result['chars']}")
                print(f"    - Слов: {result['words']}")
                print(f"    - Токенов: {result['tokens']}")
                print(f"    - Стоимость: ${cost:.6f}")
            
            print(f"\n💰 Общая стоимость тестирования: ${total_cost:.6f}")
            
            print("\n" + "=" * 70)
            print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО")
            print("=" * 70)
            print(f"\nВсе результаты сохранены в: {log_file}")
            print("Вы можете изучить их в любое время!")
            
            return 0
        else:
            print("\n❌ Не удалось получить результаты")
            return 1
            
    except Exception as e:
        print(f"\n❌ ОШИБКА при тестировании: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
