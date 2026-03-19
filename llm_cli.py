import os
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_available_models():
    return {
        "zai-org/GLM-4.7-Flash": {
            "name": "GLM-4.7-Flash",
            "prompt_price": 0.0,
            "completion_price": 0.0,
            "description": "Облачная модель GLM-4.7-Flash от Cloud.ru"
        }
    }

def calculate_cost(prompt_tokens, completion_tokens, model_name):
    models = get_available_models()
    if model_name not in models:
        model_name = "zai-org/GLM-4.7-Flash"
    
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

def get_available_modes(model_name="zai-org/GLM-4.7-Flash"):
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
                "max_completion_tokens": 100,
                "temperature": 0.7
            },
            "system_prompt": "Отвечай предельно кратко и лаконично. Максимум 50 слов.",
            "metadata": {"mode": "length_limited", "max_completion_tokens": 100, "max_words": 50}
        },
        {
            "id": 4,
            "name": "С явной инструкцией завершения",
            "params": {
                "model": model_name,
                "messages": [],
                "temperature": 0.7
            },
            "system_prompt": "После завершения ответа обязательно добавь маркер [КОНЕЦ]. Это важно для обозначения конца ответа.",
            "metadata": {"mode": "with_end_marker", "instruction": "Add [КОНЕЦ] marker"}
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
        },
        {
            "id": 6,
            "name": "С метапромптингом (двухэтапный)",
            "params": {
                "model": model_name,
                "messages": [],
                "temperature": 0.7
            },
            "metadata": {"mode": "meta_prompting", "two_stage": True}
        }
    ]

def count_words(text):
    return len(text.split())

def execute_mode(client, mode, user_input):
    # Режим 6: метапромптинг с двумя этапами
    if mode.get('metadata', {}).get('two_stage'):
        return execute_meta_prompting(client, mode, user_input)
    
    params = mode['params'].copy()
    messages = [{"role": "user", "content": user_input}]
    
    if 'system_prompt' in mode:
        messages.insert(0, {"role": "system", "content": mode['system_prompt']})
    
    params['messages'] = messages
    
    response = client.chat.completions.create(**params)
    return response

def execute_meta_prompting(client, mode, user_input):
    params = mode['params'].copy()
    
    # Этап 1: Определение роли и навыков
    meta_prompt = f"""Проанализируй следующий вопрос и определи:
1. Какая роль/профессия лучше всего ответит на этот вопрос?
2. Какая предметная область затрагивается?
3. Какие ключевые навыки и знания нужны для ответа?

Вопрос: {user_input}

Ответь кратко в формате:
РОЛЬ: [роль]
ОБЛАСТЬ: [предметная область]
НАВЫКИ: [список навыков через запятую]"""
    
    params['messages'] = [{"role": "user", "content": meta_prompt}]
    meta_response = client.chat.completions.create(**params)
    meta_analysis = meta_response.choices[0].message.content
    
    # Этап 2: Ответ с учетом определенной роли
    enriched_prompt = f"""Ты - эксперт со следующими характеристиками:

{meta_analysis}

Используя эти знания и навыки, ответь на вопрос максимально качественно и профессионально:

{user_input}"""
    
    params['messages'] = [{"role": "user", "content": enriched_prompt}]
    final_response = client.chat.completions.create(**params)
    
    # Объединяем usage из обоих запросов
    combined_usage = type('obj', (object,), {
        'prompt_tokens': meta_response.usage.prompt_tokens + final_response.usage.prompt_tokens,
        'completion_tokens': meta_response.usage.completion_tokens + final_response.usage.completion_tokens,
        'total_tokens': meta_response.usage.total_tokens + final_response.usage.total_tokens
    })
    
    # Добавляем метаанализ в начало ответа
    enriched_content = f"[МЕТААНАЛИЗ]\n{meta_analysis}\n\n[ОТВЕТ]\n{final_response.choices[0].message.content}"
    
    # Создаем объект ответа с обогащенным контентом
    final_response.choices[0].message.content = enriched_content
    final_response.usage = combined_usage
    
    return final_response

def compare_formatting_modes(client, user_input, log_file, model_name="zai-org/GLM-4.7-Flash"):
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
            print("\n[Конец ответа]")
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

def compare_reasoning_approaches(client, user_input, log_file, model_name="zai-org/GLM-4.7-Flash"):
    print("\n" + "=" * 70)
    print("ДЕНЬ 3: СРАВНЕНИЕ СПОСОБОВ РАССУЖДЕНИЯ")
    print(f"Используемая модель: {get_available_models()[model_name]['name']}")
    print(f"Задача: {user_input}")
    print("=" * 70)
    
    results = []
    
    # Способ 1: Прямой ответ (без дополнительных инструкций)
    print(f"\n{'─' * 70}")
    print("СПОСОБ 1: Прямой ответ (без дополнительных инструкций)")
    print(f"{'─' * 70}")
    
    try:
        params = {"model": model_name, "messages": [{"role": "user", "content": user_input}], "temperature": 0.7}
        response = client.chat.completions.create(**params)
        answer1 = response.choices[0].message.content
        usage1 = response.usage
        
        print(f"\nОтвет: {answer1}")
        print("\n[Конец ответа]")
        print(f"\nСтатистика: Токены: {usage1.total_tokens} | Слова: {count_words(answer1)}")
        
        results.append({
            "method": "Прямой ответ",
            "response": answer1,
            "tokens": usage1.total_tokens,
            "words": count_words(answer1),
            "chars": len(answer1)
        })
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
    
    # Способ 2: С инструкцией "решай пошагово"
    print(f"\n{'─' * 70}")
    print("СПОСОБ 2: С инструкцией 'решай пошагово'")
    print(f"{'─' * 70}")
    
    try:
        step_by_step_prompt = f"{user_input}\n\nРешай пошагово."
        params = {"model": model_name, "messages": [{"role": "user", "content": step_by_step_prompt}], "temperature": 0.7}
        response = client.chat.completions.create(**params)
        answer2 = response.choices[0].message.content
        usage2 = response.usage
        
        print(f"\nОтвет: {answer2}")
        print("\n[Конец ответа]")
        print(f"\nСтатистика: Токены: {usage2.total_tokens} | Слова: {count_words(answer2)}")
        
        results.append({
            "method": "Пошаговое решение",
            "response": answer2,
            "tokens": usage2.total_tokens,
            "words": count_words(answer2),
            "chars": len(answer2)
        })
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
    
    # Способ 3: Метапромпт (сначала создаем промпт, потом решаем)
    print(f"\n{'─' * 70}")
    print("СПОСОБ 3: Метапромпт (сначала создание промпта, затем решение)")
    print(f"{'─' * 70}")
    
    try:
        meta_request = f"Задача: {user_input}\n\nСоставь оптимальный промпт для решения этой задачи. Выведи только промпт, без дополнительных объяснений."
        params = {"model": model_name, "messages": [{"role": "user", "content": meta_request}], "temperature": 0.7}
        meta_response = client.chat.completions.create(**params)
        generated_prompt = meta_response.choices[0].message.content
        
        print(f"\nСгенерированный промпт: {generated_prompt}")
        print(f"\n{'·' * 70}")
        
        params = {"model": model_name, "messages": [{"role": "user", "content": generated_prompt}], "temperature": 0.7}
        response = client.chat.completions.create(**params)
        answer3 = response.choices[0].message.content
        usage3_combined = type('obj', (object,), {
            'total_tokens': meta_response.usage.total_tokens + response.usage.total_tokens
        })
        
        print(f"\nОтвет: {answer3}")
        print("\n[Конец ответа]")
        print(f"\nСтатистика: Токены: {usage3_combined.total_tokens} | Слова: {count_words(answer3)}")
        
        results.append({
            "method": "Метапромпт",
            "response": f"[Промпт: {generated_prompt}]\n\n{answer3}",
            "tokens": usage3_combined.total_tokens,
            "words": count_words(answer3),
            "chars": len(answer3)
        })
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
    
    # Способ 4: Группа экспертов
    print(f"\n{'─' * 70}")
    print("СПОСОБ 4: Группа экспертов (аналитик, инженер, критик)")
    print(f"{'─' * 70}")
    
    try:
        experts = [
            ("Аналитик", "Ты - аналитик. Проанализируй задачу и предложи решение с точки зрения анализа данных и логики."),
            ("Инженер", "Ты - инженер. Реши задачу с технической и практической точки зрения."),
            ("Критик", "Ты - критик. Оцени задачу критически, найди возможные проблемы и предложи решение.")
        ]
        
        expert_answers = []
        total_tokens_experts = 0
        
        for expert_name, expert_role in experts:
            print(f"\n[{expert_name}]")
            messages = [
                {"role": "system", "content": expert_role},
                {"role": "user", "content": user_input}
            ]
            params = {"model": model_name, "messages": messages, "temperature": 0.7}
            response = client.chat.completions.create(**params)
            expert_answer = response.choices[0].message.content
            total_tokens_experts += response.usage.total_tokens
            
            print(f"{expert_answer}")
            expert_answers.append(f"[{expert_name}]: {expert_answer}")
        
        combined_answer = "\n\n".join(expert_answers)
        
        print("\n[Конец ответа]")
        print(f"\nСтатистика: Токены: {total_tokens_experts} | Слова: {count_words(combined_answer)}")
        
        results.append({
            "method": "Группа экспертов",
            "response": combined_answer,
            "tokens": total_tokens_experts,
            "words": count_words(combined_answer),
            "chars": len(combined_answer)
        })
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
    
    # Сводка сравнения
    print("\n" + "=" * 70)
    print("СВОДКА СРАВНЕНИЯ СПОСОБОВ РАССУЖДЕНИЯ")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result['method']}:")
        print(f"  Символов: {result['chars']}")
        print(f"  Слов: {result['words']}")
        print(f"  Токенов: {result['tokens']}")
    
    if results:
        print("\n" + "─" * 70)
        print("АНАЛИЗ:")
        min_tokens = min(r['tokens'] for r in results)
        max_tokens = max(r['tokens'] for r in results)
        min_words = min(r['words'] for r in results)
        max_words = max(r['words'] for r in results)
        print(f"  Разброс по токенам: {min_tokens} - {max_tokens}")
        print(f"  Разброс по словам: {min_words} - {max_words}")
        print(f"  Самый краткий: {min(results, key=lambda x: x['words'])['method']}")
        print(f"  Самый подробный: {max(results, key=lambda x: x['words'])['method']}")
    
    return results

def interactive_mode_selection(client, user_input, log_file, mode_id, model_name="zai-org/GLM-4.7-Flash"):
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
        print(f"\nОтправка запроса к API...")
        print(f"Модель: {selected_mode['params']['model']}")
        response = execute_mode(client, selected_mode, user_input)
        print(f"Ответ получен!")
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
        print(f"\n❌ Ошибка при выполнении запроса:")
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Сообщение: {str(e)}")
        import traceback
        print(f"\nПолный traceback:")
        traceback.print_exc()
        return None

def main():
    api_key = os.getenv("CLOUD_API_KEY")
    
    if not api_key:
        print("Error: CLOUD_API_KEY not found in environment variables")
        print("Please create .env file with your API key")
        return
    
    base_url = "https://foundation-models.api.cloud.ru/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)
    log_file = setup_logging()
    
    print("LLM CLI Utility (Optimized for cost efficiency)")
    print("\nДоступные команды:")
    print("  'quit' / 'exit' / 'q' - выход")
    print("  'compare' - сравнение всех режимов")
    print("  'reasoning' - сравнение способов рассуждения (День 3)")
    print("  'modes' - показать список режимов")
    print("  'mode N' - переключиться на режим N (1-6)")
    print("  'models' - показать список моделей")
    print("  'model <name>' - переключиться на модель")
    print("\nТекущий режим: Стандартный (без ограничений)")
    print("Текущая модель: GLM-4.7-Flash (Cloud.ru)")
    print(f"Логи сохраняются в: {log_file}")
    print("-" * 50)
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    current_mode_id = 1
    current_model = "zai-org/GLM-4.7-Flash"
    
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
            task_input = input("Введите запрос для сравнения: ").strip()
            if task_input:
                compare_formatting_modes(client, task_input, log_file, current_model)
            continue
        
        if user_input.lower() == 'reasoning':
            task_input = input("Введите задачу для сравнения способов рассуждения: ").strip()
            if task_input:
                compare_reasoning_approaches(client, task_input, log_file, current_model)
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
