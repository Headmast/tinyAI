"""
News Agent CLI — AI-агент для автоматической генерации новостных постов.

Команды:
  generate <тема>          — создать пост через 5-шаговый pipeline
  generate -t <тип> <тема> — с указанием типа (breaking/analysis/digest/social/press)
  agent <тема>             — автономный ReAct-агент
  batch <файл>             — пакетная генерация из файла тем
  history [n]              — последние n постов (по умолчанию 10)
  export <id> <формат>     — экспортировать пост (md/html/telegram/json/plain)
  template list            — список типов постов
  template show <тип>      — показать описание типа
  models                   — список доступных моделей
  model <name>             — переключить модель
  quit / exit / q          — выход
"""

import os
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from news_agent.storage import PostStorage
from news_agent.pipeline import NewsPipeline
from news_agent.agent import AgentLoop
from news_agent.roles import POST_TYPE_GUIDES

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
            "prompt_price": 0.0002,
            "completion_price": 0.00125,
            "cached_prompt_price": 0.00002,
            "description": "Компактная версия GPT-5 от OpenAI",
            "provider": "openai",
            "id": 3
        },
        "gpt-5.4": {
            "name": "GPT-5.4",
            "prompt_price": 0.0025,
            "completion_price": 0.015,
            "cached_prompt_price": 0.00025,
            "description": "Новейшая модель GPT-5, высокая производительность",
            "provider": "openai",
            "id": 4
        },
        "gpt-5.4-mini": {
            "name": "GPT-5.4 Mini",
            "prompt_price": 0.00075,
            "completion_price": 0.0045,
            "cached_prompt_price": 0.00008,
            "description": "Облегченная версия GPT-5.4, оптимальная для большинства задач",
            "provider": "openai",
            "id": 5
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
            # Для gpt-5-nano не делаем второй запрос (игнорирует max_completion_tokens)
            model_name = mode['params']['model']
            if model_name == "gpt-5-nano":
                # Оцениваем токены на основе длины ответа
                estimated_prompt_tokens = sum(len(m['content']) for m in messages) // 4
                estimated_completion_tokens = (len(answer_text) + len(reasoning_text)) // 4
                
                class MockResponse:
                    def __init__(self, content, usage, reasoning=""):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': content,
                                'reasoning_content': reasoning
                            })()
                        })()]
                        self.usage = usage
                
                mock_usage = type('obj', (object,), {
                    'prompt_tokens': estimated_prompt_tokens,
                    'completion_tokens': estimated_completion_tokens,
                    'total_tokens': estimated_prompt_tokens + estimated_completion_tokens
                })()
                
                return MockResponse(answer_text, mock_usage, reasoning_text)
            
            # Для других моделей делаем запрос для статистики
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
        
        # Используем streaming для показа размышлений
        answer1, reasoning1 = stream_response(client, params.copy())
        
        # Оценка токенов
        estimated_tokens = (len(answer1) + len(reasoning1)) // 4
        usage1 = type('obj', (object,), {
            'total_tokens': estimated_tokens
        })()
        
        print(f"\n\n[Конец ответа]")
        print(f"Ответ: {len(answer1)} символов")
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
        
        # Используем streaming для показа размышлений
        answer2, reasoning2 = stream_response(client, params.copy())
        
        # Оценка токенов
        estimated_tokens = (len(answer2) + len(reasoning2)) // 4
        usage2 = type('obj', (object,), {
            'total_tokens': estimated_tokens
        })()
        
        print(f"\n\n[Конец ответа]")
        print(f"Ответ: {len(answer2)} символов")
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
        
        # Streaming для создания промпта
        generated_prompt, reasoning_meta = stream_response(client, params.copy())
        
        print(f"\n\nСгенерированный промпт: {generated_prompt}")
        print(f"\n{'·' * 70}")
        
        base_params = {"model": model_name, "messages": [{"role": "user", "content": generated_prompt}], "max_completion_tokens": 120000, "temperature": 0.7}
        params = get_model_params(model_name, base_params)
        print("\n🤔 Решаю по промпту...", flush=True)
        
        # Streaming для решения по промпту
        answer3, reasoning3 = stream_response(client, params.copy())
        
        # Оценка токенов
        estimated_tokens = (len(generated_prompt) + len(reasoning_meta) + len(answer3) + len(reasoning3)) // 4
        usage3_combined = type('obj', (object,), {
            'total_tokens': estimated_tokens
        })
        
        print(f"\n\n[Конец ответа]")
        print(f"Ответ: {len(answer3)} символов")
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
            
            # Используем streaming для показа размышлений эксперта
            expert_answer, expert_reasoning = stream_response(client, params.copy())
            
            # Оценка токенов
            estimated_tokens = (len(expert_answer) + len(expert_reasoning)) // 4
            total_tokens_experts += estimated_tokens
            
            print(f"\n(Токены: ~{estimated_tokens})", flush=True)
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

def compare_temperatures(client, user_input, log_file, model_name="zai-org/GLM-4.7-Flash"):
    """Сравнение влияния температуры на ответы модели"""
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ ВЛИЯНИЯ ТЕМПЕРАТУРЫ НА ОТВЕТЫ")
    print(f"Используемая модель: {get_available_models()[model_name]['name']}")
    print(f"Задача: {user_input}")
    print("=" * 70)
    
    # Проверка поддержки температуры
    if model_name == "gpt-5-nano":
        print("\n⚠️ ВНИМАНИЕ: Модель gpt-5-nano НЕ поддерживает параметр temperature!")
        print("Переключитесь на GPT-5.4, GLM-4.7-Flash или GLM-4.7 для сравнения температур.")
        print("\nИспользуйте команду: model gpt-5.4")
        return []
    
    # Для OpenAI моделей используем диапазон 0-1 с 4 точками
    # Для GLM моделей используем расширенный диапазон 0-1.2 с 3 точками
    models = get_available_models()
    provider = models[model_name]["provider"]
    
    if provider == "openai":
        temperatures = [0, 0.5, 0.75, 1.0]
        print("\n📊 Диапазон температуры для OpenAI: 0-1 (4 точки измерения)")
    else:
        temperatures = [0, 0.5, 0.75, 1.0]
        print("\n📊 Диапазон температуры: 0-1 (4 точки измерения)")
    
    results = []
    
    for temp in temperatures:
        print(f"\n{'─' * 70}")
        print(f"TEMPERATURE = {temp}")
        print(f"{'─' * 70}")
        
        # Описание характеристик температуры
        if temp == 0:
            print("Характеристики: Детерминированность, точность, повторяемость")
        elif temp == 0.5:
            print("Характеристики: Умеренная вариативность, надежность")
        elif temp == 0.75:
            print("Характеристики: Сбалансированная креативность и точность")
        elif temp == 1.0:
            print("Характеристики: Максимальная креативность в безопасном диапазоне")
        
        try:
            base_params = {
                "model": model_name,
                "messages": [{"role": "user", "content": user_input}],
                "max_completion_tokens": 120000,
                "temperature": temp
            }
            params = get_model_params(model_name, base_params)
            
            print("\n🤔 Генерирую ответ...", flush=True)
            answer, reasoning = stream_response(client, params.copy())
            
            # Оценка токенов
            estimated_tokens = (len(answer) + len(reasoning)) // 4
            word_count = count_words(answer)
            
            print(f"\n\n[Конец ответа]")
            print(f"\nСтатистика:")
            print(f"  Символов: {len(answer)}")
            print(f"  Слов: {word_count}")
            print(f"  Токенов (примерно): {estimated_tokens}")
            
            # Логирование
            usage_info = {
                "prompt_tokens": len(user_input) // 4,
                "completion_tokens": estimated_tokens,
                "total_tokens": len(user_input) // 4 + estimated_tokens,
                "word_count": word_count,
                "cost": calculate_cost(len(user_input) // 4, estimated_tokens, model_name)
            }
            
            metadata = {
                "experiment": "temperature_comparison",
                "temperature": temp
            }
            
            log_interaction(log_file, user_input, answer, usage_info, metadata)
            
            results.append({
                "temperature": temp,
                "response": answer,
                "chars": len(answer),
                "words": word_count,
                "tokens": estimated_tokens
            })
            
        except Exception as e:
            print(f"\n❌ Ошибка при temperature={temp}: {str(e)}")
    
    # Сводка и анализ
    print("\n" + "=" * 70)
    print("СВОДКА СРАВНЕНИЯ ТЕМПЕРАТУР")
    print("=" * 70)
    
    for result in results:
        print(f"\nTemperature = {result['temperature']}:")
        print(f"  Символов: {result['chars']}")
        print(f"  Слов: {result['words']}")
        print(f"  Токенов: {result['tokens']}")
    
    if results:
        print("\n" + "─" * 70)
        print("АНАЛИЗ И ВЫВОДЫ")
        print("─" * 70)
        
        # Статистика по метрикам
        chars_data = [r['chars'] for r in results]
        words_data = [r['words'] for r in results]
        
        print(f"\n📊 Разброс метрик:")
        print(f"  Символы: {min(chars_data)} - {max(chars_data)} (разница: {max(chars_data) - min(chars_data)})")
        print(f"  Слова: {min(words_data)} - {max(words_data)} (разница: {max(words_data) - min(words_data)})")
        
        # Выводы и рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ:")
        print(f"\n🎯 Temperature = 0 (Детерминированная):")
        print(f"   ✓ Задачи требующие точности и повторяемости")
        print(f"   ✓ Технические объяснения, документация")
        print(f"   ✓ Извлечение фактов, классификация")
        print(f"   ✓ Когда нужен один правильный ответ")
        
        print(f"\n📐 Temperature = 0.5 (Умеренная):")
        print(f"   ✓ Задачи требующие точности с небольшой вариативностью")
        print(f"   ✓ Деловая переписка, официальные документы")
        print(f"   ✓ Ответы на FAQ, база знаний")
        print(f"   ✓ Когда нужна стабильность, но не абсолютная")
        
        print(f"\n⚖️ Temperature = 0.75 (Сбалансированная):")
        print(f"   ✓ Универсальный вариант для большинства задач")
        print(f"   ✓ Диалоги, консультации, вопросы-ответы")
        print(f"   ✓ Объяснения с примерами и аналогиями")
        print(f"   ✓ Когда нужен баланс точности и естественности")
        
        print(f"\n🎨 Temperature = 1.0 (Креативная):")
        print(f"   ✓ Творческие задачи (рассказы, идеи, концепции)")
        print(f"   ✓ Brainstorming, генерация альтернатив")
        print(f"   ✓ Маркетинговые тексты, контент-маркетинг")
        print(f"   ✓ Когда нужно разнообразие и оригинальность")
        
        print(f"\n⚠️ ВАЖНО:")
        print(f"   • Диапазон 0-1 обеспечивает стабильное качество")
        print(f"   • Низкая температура (0-0.3) для критичных задач")
        print(f"   • Средняя температура (0.5-0.75) для общих задач")
        print(f"   • Высокая температура (0.8-1.0) для креативных задач")
    
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

def _cmd_generate(
    topic: str,
    post_type: Optional[str],
    client,
    model: str,
    storage: PostStorage,
) -> None:
    """Генерирует пост через pipeline и сохраняет."""
    pipeline = NewsPipeline(client=client, model=model, verbose=True)
    try:
        result = pipeline.run(topic=topic, post_type=post_type)
    except Exception as e:
        print(f"\n❌ Ошибка pipeline: {e}")
        import traceback; traceback.print_exc()
        return

    result["model"] = model
    post_id = storage.save(result)

    print(f"\n✅ Пост сохранён  ID: {post_id}")
    print(f"   Файл: posts/{post_id}/post.md")

    seo = result.get("seo", {})
    best_headline = (seo.get("headline_variants") or [{}])[0].get("text", "")
    if best_headline:
        print(f"   Заголовок: {best_headline}")

    meta = seo.get("meta_description", "")
    if meta:
        print(f"   Meta: {meta[:80]}...")


def _cmd_agent(topic: str, client, model: str, storage: PostStorage) -> None:
    """Запускает автономный ReAct-агент."""
    agent = AgentLoop(client=client, model=model, storage=storage, verbose=True)
    try:
        result = agent.run(task=topic)
    except Exception as e:
        print(f"\n❌ Ошибка агента: {e}")
        import traceback; traceback.print_exc()
        return

    post_id = result.get("saved_post_id")
    iterations = result.get("iterations", "?")
    tokens = result.get("token_usage", {}).get("total_tokens", "?")

    print(f"\n✅ Агент завершил работу")
    print(f"   Итераций: {iterations}  |  Токенов ~: {tokens}")
    if post_id:
        print(f"   Сохранён пост ID: {post_id}  →  posts/{post_id}/post.md")
    print(f"\n{result['final_post']}")


def _cmd_batch(filepath: str, client, model: str, storage: PostStorage) -> None:
    """Пакетная генерация из файла (одна тема = одна строка)."""
    path = Path(filepath)
    if not path.exists():
        print(f"❌ Файл не найден: {filepath}")
        return

    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"📋 Найдено тем: {len(lines)}")

    pipeline = NewsPipeline(client=client, model=model, verbose=False)
    success, failed = 0, 0

    for i, topic in enumerate(lines, 1):
        print(f"\n[{i}/{len(lines)}] {topic[:70]}")
        try:
            result = pipeline.run(topic=topic)
            result["model"] = model
            post_id = storage.save(result)
            print(f"  ✓ ID: {post_id}  слов: {len(result['post'].split())}")
            success += 1
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            failed += 1

    print(f"\n📊 Итого: {success} успешно, {failed} с ошибками")


def _cmd_history(n: int, storage: PostStorage) -> None:
    """Показывает последние n постов."""
    posts = storage.list_posts(n=n)
    if not posts:
        print("История пуста.")
        return
    print(f"\n{'─' * 65}")
    print(f"  ПОСЛЕДНИЕ ПОСТЫ ({len(posts)})")
    print(f"{'─' * 65}")
    for p in posts:
        date = p["created_at"][:10]
        tags = ", ".join(p.get("tags", [])[:3]) or "—"
        print(f"  [{p['id']}]  {date}  [{p['post_type']:8}]  {p['title'][:45]}")
        print(f"           слов: {p.get('word_count', '?')}  теги: {tags}")
    print(f"{'─' * 65}")


def _cmd_export(post_id: str, fmt: str, storage: PostStorage) -> None:
    """Экспортирует пост в указанном формате."""
    content = storage.export_post(post_id, fmt=fmt)
    if content is None:
        print(f"❌ Пост с ID '{post_id}' не найден")
        return
    out_file = Path("posts") / post_id / f"post.{fmt}"
    out_file.write_text(content, encoding="utf-8")
    print(f"✅ Экспортировано: {out_file}")
    print(f"\n--- Предпросмотр ({fmt}) ---")
    print(content[:600])
    if len(content) > 600:
        print(f"\n... ({len(content)} символов, полный файл: {out_file})")


def _cmd_template(args: list) -> None:
    """Управление шаблонами типов постов."""
    sub = args[0] if args else "list"

    if sub == "list":
        print(f"\n{'─' * 50}")
        print("  ТИПЫ ПОСТОВ")
        print(f"{'─' * 50}")
        for key, guide in POST_TYPE_GUIDES.items():
            wmin, wmax = guide["word_count"]
            print(f"  {key:12} — {guide['description']} ({wmin}-{wmax} слов, тон: {guide['tone']})")
        print(f"{'─' * 50}")

    elif sub == "show" and len(args) >= 2:
        key = args[1]
        if key not in POST_TYPE_GUIDES:
            print(f"❌ Тип '{key}' не найден. Доступны: {', '.join(POST_TYPE_GUIDES)}")
            return
        guide = POST_TYPE_GUIDES[key]
        print(f"\n[{key}] {guide['description']}")
        print(f"  Объём:     {guide['word_count'][0]}-{guide['word_count'][1]} слов")
        print(f"  Тон:       {guide['tone']}")
        print(f"  Структура: {' → '.join(guide['structure'])}")
    else:
        print("Использование: template list | template show <тип>")


def _print_help() -> None:
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              NEWS AGENT CLI  —  Команды                      ║
╠═══════════════════════════════════════════════════════════════╣
║  generate <тема>              Создать пост (pipeline)        ║
║  generate -t <тип> <тема>     С указанием типа               ║
║    типы: breaking, analysis, digest, social, press           ║
║                                                               ║
║  agent <тема>                 Автономный ReAct-агент         ║
║                                                               ║
║  batch <файл>                 Пакетная генерация из файла    ║
║                                                               ║
║  history [n]                  Последние n постов (def 10)    ║
║  export <id> <формат>         Экспорт поста                  ║
║    форматы: md, html, telegram, json, plain                  ║
║                                                               ║
║  template list                Список типов постов            ║
║  template show <тип>          Описание типа                  ║
║                                                               ║
║  models                       Список моделей                 ║
║  model <name>                 Переключить модель             ║
║                                                               ║
║  help / ?                     Эта справка                    ║
║  quit / exit / q              Выход                          ║
╚═══════════════════════════════════════════════════════════════╝""")


def main():
    cloud_api_key = os.getenv("CLOUD_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not cloud_api_key and not openai_api_key:
        print("❌ Ни CLOUD_API_KEY, ни OPENAI_API_KEY не найдены в .env")
        return

    clients = {}
    if cloud_api_key:
        clients["cloud_ru"] = OpenAI(
            api_key=cloud_api_key,
            base_url="https://foundation-models.api.cloud.ru/v1",
            timeout=120.0,
        )
    if openai_api_key:
        clients["openai"] = OpenAI(api_key=openai_api_key, timeout=120.0)

    storage = PostStorage(base_dir="posts")
    current_model = "zai-org/GLM-4.7-Flash" if "cloud_ru" in clients else "gpt-5-nano"

    print("╔══════════════════════════════════════╗")
    print("║       NEWS AGENT  v1.0               ║")
    print("║  AI-агент для генерации новостей     ║")
    print("╚══════════════════════════════════════╝")
    print(f"  Модель:     {get_available_models()[current_model]['name']}")
    print(f"  Хранилище:  posts/")
    print("  Введите 'help' для списка команд")
    print()

    session_posts = 0

    while True:
        try:
            raw = input("news-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            print(f"\n📊 Сессия: создано постов — {session_posts}")
            print("До свидания!")
            break

        elif cmd in ("help", "?"):
            _print_help()

        elif cmd == "models":
            models = get_available_models()
            print(f"\n{'─' * 55}")
            for mid, info in models.items():
                marker = "★" if mid == current_model else " "
                avail = "✓" if info["provider"] in clients else "✗"
                print(f" {avail}{marker} {info['name']:20} {mid}")
            print(f"{'─' * 55}")

        elif cmd == "model" and len(parts) >= 2:
            name = " ".join(parts[1:])
            if name in get_available_models():
                info = get_available_models()[name]
                if info["provider"] not in clients:
                    print(f"❌ Провайдер '{info['provider']}' недоступен (нет ключа API)")
                else:
                    current_model = name
                    print(f"✓ Модель: {info['name']}")
            else:
                print(f"❌ Модель '{name}' не найдена. Используйте 'models'")

        elif cmd == "generate":
            post_type = None
            topic_parts = parts[1:]
            if len(topic_parts) >= 3 and topic_parts[0] == "-t":
                post_type = topic_parts[1]
                topic_parts = topic_parts[2:]
            topic = " ".join(topic_parts)
            if not topic:
                topic = input("Тема поста: ").strip()
            if not topic:
                print("❌ Тема не может быть пустой")
                continue
            client = get_client_for_model(current_model, clients)
            _cmd_generate(topic, post_type, client, current_model, storage)
            session_posts += 1

        elif cmd == "agent":
            topic = " ".join(parts[1:])
            if not topic:
                topic = input("Задача для агента: ").strip()
            if not topic:
                print("❌ Задача не может быть пустой")
                continue
            client = get_client_for_model(current_model, clients)
            _cmd_agent(topic, client, current_model, storage)
            session_posts += 1

        elif cmd == "batch":
            if len(parts) < 2:
                print("Использование: batch <файл>")
                continue
            client = get_client_for_model(current_model, clients)
            _cmd_batch(parts[1], client, current_model, storage)

        elif cmd == "history":
            n = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 10
            _cmd_history(n, storage)

        elif cmd == "export":
            if len(parts) < 3:
                print("Использование: export <id> <формат>")
                continue
            _cmd_export(parts[1], parts[2], storage)

        elif cmd == "template":
            _cmd_template(parts[1:])

        else:
            print(f"❓ Неизвестная команда: '{cmd}'. Введите 'help'")


if __name__ == "__main__":
    main()
