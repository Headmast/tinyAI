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
            "description": "Быстрая облачная модель от Cloud.ru",
            "provider": "cloud_ru",
            "id": 1
        },
        "zai-org/GLM-4.7": {
            "name": "GLM-4.7",
            "prompt_price": 0.0,
            "completion_price": 0.0,
            "description": "Более мощная модель от Cloud.ru",
            "provider": "cloud_ru",
            "id": 2
        },
        "gpt-5-nano": {
            "name": "GPT-5 Nano",
            "prompt_price": 0.00005,
            "completion_price": 0.0004,
            "description": "Компактная версия GPT-5 от OpenAI",
            "provider": "openai",
            "id": 3
        }
    }

def get_client_for_model(model_name, clients):
    """Возвращает соответствующий client в зависимости от провайдера модели"""
    models = get_available_models()
    if model_name not in models:
        model_name = "zai-org/GLM-4.7-Flash"
    
    provider = models[model_name]["provider"]
    return clients[provider]

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

def get_model_params(model_name, base_params):
    """Адаптирует параметры под конкретную модель"""
    models = get_available_models()
    if model_name not in models:
        model_name = "zai-org/GLM-4.7-Flash"
    
    provider = models[model_name]["provider"]
    params = base_params.copy()
    
    # gpt-5-nano не поддерживает temperature != 1
    if model_name == "gpt-5-nano" and "temperature" in params:
        del params["temperature"]
    
    return params

def get_available_modes(model_name="zai-org/GLM-4.7-Flash"):
    return [
        {
            "id": 1,
            "name": "Без ограничений",
            "params": {
                "model": model_name,
                "messages": [],
                "max_completion_tokens": 120000,
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
                "max_completion_tokens": 120000,
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
                "max_completion_tokens": 120000,
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
                "max_completion_tokens": 120000,
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
                "max_completion_tokens": 120000,
                "temperature": 0.7
            },
            "metadata": {"mode": "meta_prompting", "two_stage": True}
        }
    ]

def count_words(text):
    return len(text.split())

def stream_response(client, params, show_thinking=True):
    """Потоковый вывод ответа с поэтапной печатью и размышлениями"""
    import sys
    
    # Включаем streaming
    params['stream'] = True
    
    # Включаем thinking для GLM-4.7
    model_name = params.get('model', '')
    if 'GLM' in model_name.upper():
        if 'extra_body' not in params:
            params['extra_body'] = {}
        params['extra_body']['thinking'] = {
            'type': 'enabled',
            'clear_thinking': False  # Сохраняем размышления
        }
    
    full_response = ""
    reasoning_text = ""
    in_reasoning = False
    
    print("\n", flush=True)
    
    try:
        stream = client.chat.completions.create(**params)
        
        for chunk in stream:
            # Проверяем наличие choices
            if not chunk.choices or len(chunk.choices) == 0:
                continue
            
            delta = chunk.choices[0].delta
            
            # Вывод размышлений (reasoning_content)
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                if not in_reasoning and show_thinking:
                    print("\n💭 [Размышление]", flush=True)
                    in_reasoning = True
                reasoning_text += delta.reasoning_content
                if show_thinking:
                    print(delta.reasoning_content, end="", flush=True)
            
            # Вывод основного ответа
            if hasattr(delta, 'content') and delta.content:
                if in_reasoning and show_thinking:
                    print("\n\n📝 [Ответ]", flush=True)
                    in_reasoning = False
                content = delta.content
                full_response += content
                print(content, end="", flush=True)
        
        print("\n", flush=True)
        
        # Возвращаем полный ответ для логирования
        return full_response, reasoning_text
        
    except Exception as e:
        print(f"\n❌ Ошибка при streaming: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        # Fallback на обычный режим
        params['stream'] = False
        if 'extra_body' in params:
            del params['extra_body']
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content, ""

def execute_mode(client, mode, user_input, max_retries=3):
    # Режим 6: метапромптинг с двумя этапами
    if mode.get('metadata', {}).get('two_stage'):
        return execute_meta_prompting(client, mode, user_input, max_retries)
    
    # Адаптируем параметры под модель
    model_name = mode['params']['model']
    params = get_model_params(model_name, mode['params'])
    messages = [{"role": "user", "content": user_input}]
    
    if 'system_prompt' in mode:
        messages.insert(0, {"role": "system", "content": mode['system_prompt']})
    
    params['messages'] = messages
    
    # Retry механизм для нестабильных API
    for attempt in range(max_retries):
        try:
            print("🤔 Размышляю...", flush=True)
            
            # Используем streaming для постепенного вывода
            answer_text, reasoning_text = stream_response(client, params.copy())
            
            # Создаём объект ответа для совместимости
            # Делаем обычный запрос для получения usage статистики
            params_for_usage = params.copy()
            params_for_usage['max_completion_tokens'] = 1  # Минимальный запрос для статистики
            if 'extra_body' in params_for_usage:
                del params_for_usage['extra_body']  # Убираем thinking для статистики
            usage_response = client.chat.completions.create(**params_for_usage)
            
            # Создаём mock объект с ответом
            class MockResponse:
                def __init__(self, content, usage, reasoning=""):
                    self.choices = [type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': content,
                            'reasoning_content': reasoning
                        })()
                    })()]
                    self.usage = usage
            
            # Примерная оценка токенов (1 токен ≈ 4 символа)
            estimated_tokens = (len(answer_text) + len(reasoning_text)) // 4
            mock_usage = type('obj', (object,), {
                'prompt_tokens': usage_response.usage.prompt_tokens,
                'completion_tokens': estimated_tokens,
                'total_tokens': usage_response.usage.prompt_tokens + estimated_tokens
            })()
            
            return MockResponse(answer_text, mock_usage, reasoning_text)
            
        except Exception as e:
            print("\r", end="", flush=True)
            if attempt < max_retries - 1:
                print(f"⚠️ Попытка {attempt + 1} не удалась, повтор через 2 сек...", flush=True)
                import time
                time.sleep(2)
            else:
                raise

def execute_meta_prompting(client, mode, user_input, max_retries=3):
    model_name = mode['params']['model']
    params = get_model_params(model_name, mode['params'])
    
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
    
    # Retry для этапа 1
    for attempt in range(max_retries):
        try:
            print("🤔 Этап 1: Анализирую задачу...", flush=True)
            meta_response = client.chat.completions.create(**params)
            print("\r✓ Этап 1 завершен!        ", flush=True)
            break
        except Exception as e:
            print("\r", end="", flush=True)
            if attempt < max_retries - 1:
                print(f"⚠️ Этап 1: Попытка {attempt + 1} не удалась, повтор через 2 сек...", flush=True)
                import time
                time.sleep(2)
            else:
                raise
    
    meta_analysis = meta_response.choices[0].message.content
    
    # Этап 2: Ответ с учетом определенной роли
    enriched_prompt = f"""Ты - эксперт со следующими характеристиками:

{meta_analysis}

Используя эти знания и навыки, ответь на вопрос максимально качественно и профессионально:

{user_input}"""
    
    params['messages'] = [{"role": "user", "content": enriched_prompt}]
    
    # Retry для этапа 2
    for attempt in range(max_retries):
        try:
            print("🤔 Этап 2: Формирую экспертный ответ...", flush=True)
            final_response = client.chat.completions.create(**params)
            print("\r✓ Этап 2 завершен!              ", flush=True)
            break
        except Exception as e:
            print("\r", end="", flush=True)
            if attempt < max_retries - 1:
                print(f"⚠️ Этап 2: Попытка {attempt + 1} не удалась, повтор через 2 сек...", flush=True)
                import time
                time.sleep(2)
            else:
                raise
    
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
        base_params = {"model": model_name, "messages": [{"role": "user", "content": user_input}], "max_completion_tokens": 120000, "temperature": 0.7}
        params = get_model_params(model_name, base_params)
        print("\n🤔 Размышляю...", flush=True)
        response = client.chat.completions.create(**params)
        print("\r✓ Готово!      ", flush=True)
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
        base_params = {"model": model_name, "messages": [{"role": "user", "content": step_by_step_prompt}], "max_completion_tokens": 120000, "temperature": 0.7}
        params = get_model_params(model_name, base_params)
        print("\n🤔 Размышляю пошагово...", flush=True)
        response = client.chat.completions.create(**params)
        print("\r✓ Готово!               ", flush=True)
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
        base_params = {"model": model_name, "messages": [{"role": "user", "content": meta_request}], "max_completion_tokens": 120000, "temperature": 0.7}
        params = get_model_params(model_name, base_params)
        print("\n🤔 Создаю промпт...", flush=True)
        meta_response = client.chat.completions.create(**params)
        print("\r✓ Промпт создан!    ", flush=True)
        generated_prompt = meta_response.choices[0].message.content
        
        print(f"\nСгенерированный промпт: {generated_prompt}")
        print(f"\n{'·' * 70}")
        
        base_params = {"model": model_name, "messages": [{"role": "user", "content": generated_prompt}], "max_completion_tokens": 120000, "temperature": 0.7}
        params = get_model_params(model_name, base_params)
        print("\n🤔 Решаю по промпту...", flush=True)
        response = client.chat.completions.create(**params)
        print("\r✓ Готово!              ", flush=True)
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
            print(f"\n[{expert_name}]", flush=True)
            messages = [
                {"role": "system", "content": expert_role},
                {"role": "user", "content": user_input}
            ]
            base_params = {"model": model_name, "messages": messages, "max_completion_tokens": 120000, "temperature": 0.7}
            params = get_model_params(model_name, base_params)
            print(f"🤔 {expert_name} размышляет...", flush=True)
            response = client.chat.completions.create(**params)
            print(f"\r✓ {expert_name} ответил!        ", flush=True)
            expert_answer = response.choices[0].message.content
            total_tokens_experts += response.usage.total_tokens
            
            print(expert_answer, flush=True)
            print(f"(Токены: {response.usage.total_tokens})", flush=True)
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
    # Проверка ключей API
    cloud_api_key = os.getenv("CLOUD_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not cloud_api_key and not openai_api_key:
        print("Error: Neither CLOUD_API_KEY nor OPENAI_API_KEY found")
        print("Please create .env file with at least one API key")
        return
    
    # Создание клиентов для разных провайдеров
    clients = {}
    
    if cloud_api_key:
        cloud_url = "https://foundation-models.api.cloud.ru/v1"
        clients["cloud_ru"] = OpenAI(api_key=cloud_api_key, base_url=cloud_url, timeout=60.0)
    
    if openai_api_key:
        clients["openai"] = OpenAI(api_key=openai_api_key, timeout=60.0)
    
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
    print("Текущая модель: GPT-5 Nano (OpenAI)")
    print(f"Логи сохраняются в: {log_file}")
    print("-" * 50)
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    current_mode_id = 1
    current_model = "gpt-5-nano"
    
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
                client = get_client_for_model(current_model, clients)
                compare_formatting_modes(client, task_input, log_file, current_model)
            continue
        
        if user_input.lower() == 'reasoning':
            task_input = input("Введите задачу для сравнения способов рассуждения: ").strip()
            if task_input:
                client = get_client_for_model(current_model, clients)
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
        
        # Выбор модели перед запросом
        print("\nВыберите модель:")
        models = get_available_models()
        # Фильтруем модели по доступности провайдеров
        available_models = {k: v for k, v in models.items() if v["provider"] in clients}
        for model_key, model_info in available_models.items():
            marker = "★" if model_key == current_model else " "
            print(f"{marker} {model_info['id']}. {model_info['name']} - {model_info['description']}")
        
        model_choice = input("Модель (1, 2 или 3, Enter = текущая): ").strip()
        
        if model_choice:
            selected_model = None
            for model_key, model_info in models.items():
                if str(model_info['id']) == model_choice:
                    selected_model = model_key
                    break
            
            if selected_model:
                current_model = selected_model
                print(f"✓ Выбрана модель: {models[current_model]['name']}")
            else:
                print(f"Используется текущая модель: {models[current_model]['name']}")
        else:
            print(f"Используется текущая модель: {models[current_model]['name']}")
        
        try:
            client = get_client_for_model(current_model, clients)
            usage_info = interactive_mode_selection(client, user_input, log_file, current_mode_id, current_model)
            
            if usage_info:
                total_prompt_tokens += usage_info['prompt_tokens']
                total_completion_tokens += usage_info['completion_tokens']
                total_cost += usage_info['cost']
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
