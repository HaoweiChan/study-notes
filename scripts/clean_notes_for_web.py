import os
import re
import sys

def clean_markdown(text):
    # Regex to remove ## Flashcards and ## Quizzes sections
    # Matches ## Section ... until the next ## Header or End of String
    
    # Pattern explanation:
    # ^##\s+ : Starts with ## and whitespace
    # (Flashcards|Quizzes) : The headers we want to remove
    # .*? : Non-greedy match of content
    # (?=^##\s|\Z) : Lookahead for next header or End of File
    
    pattern = r"(?ms)^##\s+(Flashcards|Quizzes).*?(?=^##\s|\Z)"
    
    cleaned_text = re.sub(pattern, "", text)
    
    # Remove ```flashcard ... ``` blocks
    flashcard_block = r"(?ms)^```flashcard.*?^```"
    cleaned_text = re.sub(flashcard_block, "", cleaned_text)
    
    # Remove extra newlines created by deletion
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    
    return cleaned_text

def main():
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = "web/public/notes"

    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} does not exist.")
        return

    count = 0
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                new_content = clean_markdown(content)
                
                if new_content != content:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    count += 1
                    
    print(f"Cleaned {count} notes in {target_dir}")

if __name__ == "__main__":
    main()
