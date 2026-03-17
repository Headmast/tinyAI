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

def compare_formatting_modes(client, user_input, log_file):
    print("\n" + "=" * 70)
    print("РЕЖИМ СРАВНЕНИЯ: отправка одного запроса с разными параметрами")
    print("=" * 70)
    
    modes = [
        {
            "name": "Без ограничений",
            "params": {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": user_input}],
                "temperature": 0.7
            },
            "metadata": {"mode": "unrestricted"}
        },
        {
            "name": "С явным форматом ответа",
            "params": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "Отвечай в формате JSON с полями: 'answer' (основной ответ) и 'summary' (краткое резюме в 1 предложение)."},
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.7
            },
            "metadata": {"mode": "formatted", "format": "JSON with answer and summary fields"}
        },
        {
            "name": "С ограничением длины",
            "params": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "Отвечай кратко, максимум 2-3 предложения."},
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": 100,
                "temperature": 0.7
            },
            "metadata": {"mode": "length_limited", "max_tokens": 100}
        },
        {
            "name": "С условием завершения (stop sequence)",
            "params": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "Заканчивай каждый ответ словом '[КОНЕЦ]'."},
                    {"role": "user", "content": user_input}
                ],
                "stop": ["[КОНЕЦ]"],
                "temperature": 0.7
            },
            "metadata": {"mode": "with_stop_sequence", "stop": ["[КОНЕЦ]"]}
        }
    ]
    
    results = []
    
    for i, mode in enumerate(modes, 1):
        print(f"\n{'─' * 70}")
        print(f"Режим {i}: {mode['name']}")
        print(f"{'─' * 70}")
        
        try:
            response = client.chat.completions.create(**mode['params'])
            assistant_message = response.choices[0].message.content
            usage = response.usage
            
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            cost = (prompt_tokens * 0.0015 / 1000) + (completion_tokens * 0.002 / 1000)
            
            usage_info = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": cost
            }
            
            print(f"\nОтвет: {assistant_message}")
            print(f"\nТокены: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
            print(f"Стоимость: ${cost:.6f}")
            
            log_interaction(log_file, user_input, assistant_message, usage_info, mode['metadata'])
            
            results.append({
                "mode": mode['name'],
                "response": assistant_message,
                "tokens": total_tokens,
                "cost": cost
            })
            
        except Exception as e:
            print(f"\nОшибка в режиме '{mode['name']}': {str(e)}")
    
    print("\n" + "=" * 70)
    print("СВОДКА СРАВНЕНИЯ")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result['mode']}:")
        print(f"  Длина ответа: {len(result['response'])} символов")
        print(f"  Токены: {result['tokens']}")
        print(f"  Стоимость: ${result['cost']:.6f}")
    
    return results

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create .env file with your API key")
        return
    
    client = OpenAI(api_key=api_key)
    log_file = setup_logging()
    
    print("LLM CLI Utility (Optimized for cost efficiency)")
    print("Type your message (or 'quit' to exit)")
    print("Type 'compare' to enter comparison mode")
    print(f"Логи сохраняются в: {log_file}")
    print("-" * 50)
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    
    while True:
        user_input = input("\nYou: ").strip()
        
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
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": user_input}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            assistant_message = response.choices[0].message.content
            usage = response.usage
            
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            
            cost_per_request = (prompt_tokens * 0.0015 / 1000) + (completion_tokens * 0.002 / 1000)
            total_cost += cost_per_request
            
            usage_info = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": cost_per_request
            }
            
            log_interaction(log_file, user_input, assistant_message, usage_info, {"mode": "normal"})
            
            print(f"\nAssistant: {assistant_message}")
            print(f"\n[Tokens used: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens}) | Cost: ${cost_per_request:.6f}]")
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
