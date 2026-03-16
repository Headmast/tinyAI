import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create .env file with your API key")
        return
    
    client = OpenAI(api_key=api_key)
    
    print("LLM CLI Utility (Optimized for cost efficiency)")
    print("Type your message (or 'quit' to exit)")
    print("-" * 50)
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    
    while True:
        user_input = input("\nYou: ").strip()
        
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
            
            print(f"\nAssistant: {assistant_message}")
            print(f"\n[Tokens used: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens}) | Cost: ${cost_per_request:.6f}]")
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
