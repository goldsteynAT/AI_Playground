import ollama
import os

model ="phi4:latest"

input_file = "./data/grocery_list.txt"
output_file = "./data/categorized_grocery_list.txt"

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Input file {input_file} not found.")
    exit()

# Read the un-categorized grocery items from the input file
with open(input_file, "r") as f:
    items = f.readlines()

prompt = f"""
You are an assistant that categorizes grocery items. 
Here is a list of grocery items:

{items}

Please:
1. Categorize these items into appropriate categories such as fruits, vegetables, dairy, etc.
2. Sort the items alphabetically within each category.
3. Present the categorized list in a clear and organized manner.
4. Do not make any notes or comments.

"""

try:
    response = ollama.generate(model=model, prompt=prompt)
    generated_text = response.get("response", "")
    print("Categrized list:")
    print(generated_text)


    with open(output_file, "w") as f:
        f.write(generated_text.strip())
        
    print (f"Categorized grocery list saved to {output_file}")
except Exception as e:
    print("An error occured:", str(e))