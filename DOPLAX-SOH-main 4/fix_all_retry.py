with open('main_HUST.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("if invalid_experiment:\n                    continue  # retry same experiment index", 
                          "if False:\n                    continue  # retry same experiment index")

with open('main_HUST.py', 'w', encoding='utf-8') as f:
    f.write(content)
