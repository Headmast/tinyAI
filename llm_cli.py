import os
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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

def get_available_modes():
    return [
        {
            "id": 1,
            "name": "Без ограничений",
            "params": {
                "model": "gpt-3.5-turbo",
                "messages": [],
                "temperature": 0.7
            },
            "metadata": {"mode": "unrestricted"}
        },
        {
            "id": 2,
            "name": "С явным форматом ответа",
            "params": {
                "model": "gpt-3.5-turbo",
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
                "model": "gpt-3.5-turbo",
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
                "model": "gpt-3.5-turbo",
                "messages": [],
                "stop": ["[КОНЕЦ]"],
                "temperature": 0.7
            },
            "system_prompt": "После завершения ответа обязательно добавь маркер [КОНЕЦ]",
            "metadata": {"mode": "with_stop_sequence", "stop": ["[КОНЕЦ]"]}
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

def compare_formatting_modes(client, user_input, log_file):
    print("\n" + "=" * 70)
    print("РЕЖИМ СРАВНЕНИЯ: отправка одного запроса с разными параметрами")
    print("=" * 70)
    
    modes = get_available_modes()
    
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
            
            cost = (prompt_tokens * 0.0015 / 1000) + (completion_tokens * 0.002 / 1000)
            
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

def interactive_mode_selection(client, user_input, log_file, mode_id):
    modes = get_available_modes()
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
        
        cost = (prompt_tokens * 0.0015 / 1000) + (completion_tokens * 0.002 / 1000)
        
        usage_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "word_count": word_count,
            "cost": cost
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
    print("  'mode N' - переключиться на режим N (1-4)")
    print("\nТекущий режим: Стандартный (без ограничений)")
    print(f"Логи сохраняются в: {log_file}")
    print("-" * 50)
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    current_mode_id = 1
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'modes':
            print("\nДоступные режимы:")
            for mode in get_available_modes():
                marker = "★" if mode['id'] == current_mode_id else " "
                print(f"{marker} {mode['id']}. {mode['name']}")
                if 'system_prompt' in mode:
                    print(f"   Промпт: {mode['system_prompt'][:70]}...")
            continue
        
        if user_input.lower().startswith('mode '):
            try:
                mode_num = int(user_input.split()[1])
                if 1 <= mode_num <= 4:
                    current_mode_id = mode_num
                    mode_name = get_available_modes()[mode_num - 1]['name']
                    print(f"\n✓ Переключено на режим {mode_num}: {mode_name}")
                else:
                    print("Ошибка: Выберите режим от 1 до 4")
            except (IndexError, ValueError):
                print("Ошибка: Используйте формат 'mode N', где N - номер режима (1-4)")
            continue
        
        if user_input.lower() == 'compare':
            compare_input = input("Введите запрос для сравнения: ").strip()
            if compare_input:
                compare_formatting_modes(client, compare_input, log_file)
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
            usage_info = interactive_mode_selection(client, user_input, log_file, current_mode_id)
            
            if usage_info:
                total_prompt_tokens += usage_info['prompt_tokens']
                total_completion_tokens += usage_info['completion_tokens']
                total_cost += usage_info['cost']
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
