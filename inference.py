import torch
from part_0_1_contract import load_model_and_tokenizer, predict_answer

def interactive_test(checkpoint_path):
    model, tokenizer = load_model_and_tokenizer(".")
    p = tokenizer['p']
    
    print(f"Model loaded for p={p}. Enter 'q' to quit.")
    while True:
        try:
            inp = input("Enter a, b, op (e.g., 10 20 +): ")
            if inp.lower() == 'q': break
            a, b, op = inp.split()
            ans = predict_answer(model, tokenizer, int(a), int(b), op, p)
            print(f"Model Prediction: {ans}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_test("grokking_div.pt")