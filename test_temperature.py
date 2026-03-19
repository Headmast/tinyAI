#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функционала сравнения температур
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
    print("ТЕСТИРОВАНИЕ ФУНКЦИОНАЛА СРАВНЕНИЯ ТЕМПЕРАТУР")
    print("=" * 70)
    
    cloud_api_key = os.getenv("CLOUD_API_KEY")
    
    if not cloud_api_key:
        print("\n❌ ОШИБКА: CLOUD_API_KEY не найден в .env")
        print("Для тестирования температур необходим API ключ Cloud.ru")
        return 1
    
    print("\n✓ API ключ Cloud.ru найден")
    
    cloud_url = "https://foundation-models.api.cloud.ru/v1"
    clients = {
        "cloud_ru": OpenAI(api_key=cloud_api_key, base_url=cloud_url, timeout=60.0)
    }
    
    model_name = "zai-org/GLM-4.7-Flash"
    models = get_available_models()
    
    print(f"✓ Используемая модель: {models[model_name]['name']}")
    print(f"✓ Провайдер: {models[model_name]['provider']}")
    
    log_file = setup_logging()
    print(f"✓ Лог-файл создан: {log_file}")
    
    test_question = "Как лучше организовывать свой рабочий день, бороться с прокрастинацией и быть успешным?"
    
    print(f"\n✓ Тестовый вопрос: {test_question}")
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
            for result in results:
                print(f"\n  Temperature {result['temperature']}:")
                print(f"    - Символов: {result['chars']}")
                print(f"    - Слов: {result['words']}")
                print(f"    - Токенов: {result['tokens']}")
            
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
