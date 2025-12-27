import os
import json
import frontmatter

NOTES_DIR = "notes"
OUTPUT_FILE = "docs/artifacts/notes.json"

def main():
    notes_list = []
    
    for root, dirs, files in os.walk(NOTES_DIR):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, NOTES_DIR)
                
                with open(path, "r", encoding="utf-8") as f:
                    post = frontmatter.load(f)
                    
                note_data = {
                    "title": post.get("title", file),
                    "category": post.get("category", os.path.basename(root)),
                    "slug": post.get("slug", file.replace(".md", "")),
                    "path": f"notes/{rel_path}",
                    "tags": post.get("tags", []),
                    "date": str(post.get("date", ""))
                }
                notes_list.append(note_data)
                
    # Sort by date desc
    notes_list.sort(key=lambda x: x["date"], reverse=True)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(notes_list, f, indent=2)
        
    print(f"Exported {len(notes_list)} notes to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

