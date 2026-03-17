import os
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_available_models():
    return {
        "gpt-3.5-turbo": {
            "name": "GPT-3.5 Turbo",
            "prompt_price": 0.0015,
            "completion_price": 0.002,
            "description": "Быстрая и экономичная модель"
        },
        "gpt-4o-mini": {
            "name": "GPT-4o Mini",
            "prompt_price": 0.00015,
            "completion_price": 0.0006,
            "description": "Самая экономичная модель GPT-4 класса"
        },
        "gpt-4o": {
            "name": "GPT-4o",
            "prompt_price": 0.005,
            "completion_price": 0.015,
            "description": "Оптимизированная GPT-4, баланс цены и качества"
        },
        "gpt-4-turbo": {
            "name": "GPT-4 Turbo",
            "prompt_price": 0.01,
            "completion_price": 0.03,
            "description": "Быстрая версия GPT-4"
        },
        "gpt-4": {
            "name": "GPT-4",
            "prompt_price": 0.03,
            "completion_price": 0.06,
            "description": "Наиболее мощная модель, высокая стоимость"
        },
        "gpt-5.4": {
            "name": "GPT-5.4",
            "prompt_price": 0.0025,
            "completion_price": 0.015,
            "description": "Новейшая модель GPT-5, высокая производительность"
        }
    }

def calculate_cost(prompt_tokens, completion_tokens, model_name):
    models = get_available_models()
    if model_name not in models:
        model_name = "gpt-3.5-turbo"
    
    model = models[model_name]
    prompt_cost = (prompt_tokens * model["prompt_price"]) / 1000
    completion_cost = (completion_tokens * model["completion_price"]) / 1000
    return prompt_cost + completion_cost

def setup_logging():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"conversation_{timestamp}.json"
    return log_file

def log_interaction(log_file, user_input, assistant_response, usage_info, metadata=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "assistant_response": assistant_response,
        "usage": usage_info,
        "metadata": metadata or {}
    }
    
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def get_available_modes(model_name="gpt-3.5-turbo"):
    return [
        {
            "id": 1,
            "name": "Без ограничений",
            "params": {
                "model": model_name,
                "messages": [],
                "temperature": 0.7
            },
            "metadata": {"mode": "unrestricted"}
        },
        {
            "id": 2,
            "name": "С явным форматом ответа",
            "params": {
                "model": model_name,
                "messages": [],
                "temperature": 0.7
            },
            "system_prompt": "Структурируй свой ответ следующим образом:\n1. ОСНОВНОЙ ОТВЕТ: (подробный ответ на вопрос)\n2. КРАТКОЕ РЕЗЮМЕ: (1-2 предложения)\n3. КЛЮЧЕВЫЕ МОМЕНТЫ: (список из 2-3 пунктов)",
            "metadata": {"mode": "formatted", "format": "Structured text with sections"}
        },
        {
            "id": 3,
            "name": "С ограничением длины",
            "params": {
                "model": model_name,
                "messages": [],
                "max_tokens": 100,
                "temperature": 0.7
            },
            "system_prompt": "Отвечай предельно кратко и лаконично. Максимум 50 слов.",
            "metadata": {"mode": "length_limited", "max_tokens": 100, "max_words": 50}
        },
        {
            "id": 4,
            "name": "С условием завершения (stop sequence)",
            "params": {
                "model": model_name,
                "messages": [],
                "stop": ["[КОНЕЦ]"],
                "temperature": 0.7
            },
            "system_prompt": "После завершения ответа обязательно добавь маркер [КОНЕЦ]",
            "metadata": {"mode": "with_stop_sequence", "stop": ["[КОНЕЦ]"]}
        },
        {
            "id": 5,
            "name": "С выводом в формате JSON",
            "params": {
                "model": model_name,
                "messages": [],
                "temperature": 0.7
            },
            "system_prompt": "Отвечай СТРОГО в формате JSON с полями:\n- \"answer\": подробный ответ на вопрос\n- \"summary\": краткое резюме в 1-2 предложения\n- \"key_points\": массив из 2-3 ключевых моментов\nВозвращай только валидный JSON, без дополнительного текста.",
            "metadata": {"mode": "json_format", "format": "JSON with answer, summary, and key_points"}
        }
    ]

def count_words(text):
    return len(text.split())

def execute_mode(client, mode, user_input):
    params = mode['params'].copy()
    messages = [{"role": "user", "content": user_input}]
    
    if 'system_prompt' in mode:
        messages.insert(0, {"role": "system", "content": mode['system_prompt']})
    
    params['messages'] = messages
    
    response = client.chat.completions.create(**params)
    return response

def compare_formatting_modes(client, user_input, log_file, model_name="gpt-3.5-turbo"):
    print("\n" + "=" * 70)
    print("РЕЖИМ СРАВНЕНИЯ: отправка одного запроса с разными параметрами")
    print(f"Используемая модель: {get_available_models()[model_name]['name']}")
    print("=" * 70)
    
    modes = get_available_modes(model_name)
    
    results = []
    
    for mode in modes:
        print(f"\n{'─' * 70}")
        print(f"Режим {mode['id']}: {mode['name']}")
        if 'system_prompt' in mode:
            print(f"Системный промпт: {mode['system_prompt'][:60]}...")
        print(f"{'─' * 70}")
        
        try:
            response = execute_mode(client, mode, user_input)
            assistant_message = response.choices[0].message.content
            usage = response.usage
            
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            word_count = count_words(assistant_message)
            
            cost = calculate_cost(prompt_tokens, completion_tokens, mode['params']['model'])
            
            usage_info = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "word_count": word_count,
                "cost": cost
            }
            
            print(f"\nОтвет: {assistant_message}")
            print(f"\nСтатистика:")
            print(f"  Токены: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
            print(f"  Слов: {word_count}")
            print(f"  Символов: {len(assistant_message)}")
            print(f"  Стоимость: ${cost:.6f}")
            
            log_interaction(log_file, user_input, assistant_message, usage_info, mode['metadata'])
            
            results.append({
                "mode": mode['name'],
                "response": assistant_message,
                "tokens": total_tokens,
                "words": word_count,
                "chars": len(assistant_message),
                "cost": cost
            })
            
        except Exception as e:
            print(f"\nОшибка в режиме '{mode['name']}': {str(e)}")
    
    print("\n" + "=" * 70)
    print("СВОДКА СРАВНЕНИЯ")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result['mode']}:")
        print(f"  Символов: {result['chars']}")
        print(f"  Слов: {result['words']}")
        print(f"  Токенов: {result['tokens']}")
        print(f"  Стоимость: ${result['cost']:.6f}")
    
    if results:
        print("\n" + "─" * 70)
        print("АНАЛИЗ:")
        min_cost = min(r['cost'] for r in results)
        max_cost = max(r['cost'] for r in results)
        min_words = min(r['words'] for r in results)
        max_words = max(r['words'] for r in results)
        print(f"  Разброс стоимости: ${min_cost:.6f} - ${max_cost:.6f}")
        print(f"  Разброс по словам: {min_words} - {max_words}")
        print(f"  Экономия при выборе самого дешевого: ${max_cost - min_cost:.6f}")
    
    return results

def interactive_mode_selection(client, user_input, log_file, mode_id, model_name="gpt-3.5-turbo"):
    modes = get_available_modes(model_name)
    selected_mode = None
    
    for mode in modes:
        if mode['id'] == mode_id:
            selected_mode = mode
            break
    
    if not selected_mode:
        print(f"Ошибка: Режим {mode_id} не найден")
        return
    
    print(f"\nИспользуется режим: {selected_mode['name']}")
    if 'system_prompt' in selected_mode:
        print(f"Системный промпт: {selected_mode['system_prompt']}")
    
    try:
        response = execute_mode(client, selected_mode, user_input)
        assistant_message = response.choices[0].message.content
        usage = response.usage
        
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        word_count = count_words(assistant_message)
        
        cost = calculate_cost(prompt_tokens, completion_tokens, selected_mode['params']['model'])
        
        usage_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "word_count": word_count,
            "cost": cost,
            "model": selected_mode['params']['model']
        }
        
        log_interaction(log_file, user_input, assistant_message, usage_info, selected_mode['metadata'])
        
        print(f"\nAssistant: {assistant_message}")
        print(f"\n[Режим: {selected_mode['name']} | Токены: {total_tokens} | Слова: {word_count} | Стоимость: ${cost:.6f}]")
        
        return usage_info
        
    except Exception as e:
        print(f"\nОшибка: {str(e)}")
        return None

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create .env file with your API key")
        return
    
    client = OpenAI(api_key=api_key)
    log_file = setup_logging()
    
    print("LLM CLI Utility (Optimized for cost efficiency)")
    print("\nДоступные команды:")
    print("  'quit' / 'exit' / 'q' - выход")
    print("  'compare' - сравнение всех режимов")
    print("  'modes' - показать список режимов")
    print("  'mode N' - переключиться на режим N (1-5)")
    print("  'models' - показать список моделей")
    print("  'model <name>' - переключиться на модель")
    print("\nТекущий режим: Стандартный (без ограничений)")
    print("Текущая модель: GPT-3.5 Turbo")
    print(f"Логи сохраняются в: {log_file}")
    print("-" * 50)
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    current_mode_id = 1
    current_model = "gpt-3.5-turbo"
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'models':
            print("\nДоступные модели:")
            models = get_available_models()
            for model_id, model_info in models.items():
                marker = "★" if model_id == current_model else " "
                print(f"{marker} {model_id}")
                print(f"   Название: {model_info['name']}")
                print(f"   Описание: {model_info['description']}")
                print(f"   Цена: ${model_info['prompt_price']}/1K prompt, ${model_info['completion_price']}/1K completion")
            continue
        
        if user_input.lower() == 'modes':
            print("\nДоступные режимы:")
            for mode in get_available_modes(current_model):
                marker = "★" if mode['id'] == current_mode_id else " "
                print(f"{marker} {mode['id']}. {mode['name']}")
                if 'system_prompt' in mode:
                    print(f"   Промпт: {mode['system_prompt'][:70]}...")
            continue
        
        if user_input.lower().startswith('model '):
            model_name = user_input[6:].strip()
            if model_name in get_available_models():
                current_model = model_name
                model_info = get_available_models()[model_name]
                print(f"\n✓ Переключено на модель: {model_info['name']}")
                print(f"  Цена: ${model_info['prompt_price']}/1K prompt, ${model_info['completion_price']}/1K completion")
            else:
                print(f"Ошибка: Модель '{model_name}' не найдена")
                print("Используйте 'models' для просмотра доступных моделей")
            continue
        
        if user_input.lower().startswith('mode '):
            try:
                mode_num = int(user_input.split()[1])
                if 1 <= mode_num <= 5:
                    current_mode_id = mode_num
                    mode_name = get_available_modes(current_model)[mode_num - 1]['name']
                    print(f"\n✓ Переключено на режим {mode_num}: {mode_name}")
                else:
                    print("Ошибка: Выберите режим от 1 до 5")
            except (IndexError, ValueError):
                print("Ошибка: Используйте формат 'mode N', где N - номер режима (1-5)")
            continue
        
        if user_input.lower() == 'compare':
            compare_input = input("Введите запрос для сравнения: ").strip()
            if compare_input:
                compare_formatting_modes(client, compare_input, log_file, current_model)
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n" + "=" * 50)
            print(f"Session summary:")
            print(f"  Total prompt tokens: {total_prompt_tokens}")
            print(f"  Total completion tokens: {total_completion_tokens}")
            print(f"  Total tokens: {total_prompt_tokens + total_completion_tokens}")
            print(f"  Estimated cost: ${total_cost:.6f}")
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            usage_info = interactive_mode_selection(client, user_input, log_file, current_mode_id, current_model)
            
            if usage_info:
                total_prompt_tokens += usage_info['prompt_tokens']
                total_completion_tokens += usage_info['completion_tokens']
                total_cost += usage_info['cost']
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
