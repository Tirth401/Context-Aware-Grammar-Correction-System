from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model():
    # Use a pre-trained grammar correction model fine-tuned for English
    model_name = "vennify/t5-base-grammar-correction"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def correct_grammar(text, model, tokenizer):
    # Add the task-specific prefix required for the model
    input_text = "grammar: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(input_ids, max_length=128, num_return_sequences=1, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

if __name__ == "__main__":
    # Load the fine-tuned model and tokenizer
    model, tokenizer = load_model()

    # Test the grammar correction
    test_text = "I goes to the store yesterday."
    corrected = correct_grammar(test_text, model, tokenizer)

    # Display the results
    print(f"Original: {test_text}")
    print(f"Corrected: {corrected}")
