#!/usr/bin/env python3
# scripts/export_quizzes_json.py
import re
import json
import sys
from pathlib import Path

import frontmatter

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"
OUT = ROOT / "docs" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

# Regex patterns for markdown quiz sections
QUIZ_SECTION_RE = re.compile(r"^##+\s*Quizzes\s*(.*?)^(?:##\s|\Z)", re.S|re.I|re.M)
QUIZ_BLOCK_RE = re.compile(
    r"Q:\s*(.+?)\n\s*Options:\s*\n((?:\s*-\s*[A-Z]\).*?\n)+)\s*Answers?:\s*(.+?)\n\s*Explanation:\s*(.+?)(?=\n\s*Q:|$)",
    re.S|re.I
)

def extract_from_frontmatter(text):
    """Extract quizzes from frontmatter."""
    try:
        fm = frontmatter.loads(text)
    except Exception:
        return []
    
    quizzes = []
    if isinstance(fm.metadata.get("quizzes"), list):
        for item in fm.metadata["quizzes"]:
            q = item.get("q") or item.get("Q") or item.get("question")
            options = item.get("options")
            answers = item.get("answers") or item.get("answer")
            explanation = item.get("explanation") or item.get("explain")
            
            if q and options and answers is not None:
                # Ensure answers is a list
                if not isinstance(answers, list):
                    answers = [answers]
                
                quizzes.append({
                    "q": q.strip(),
                    "options": [opt.strip() for opt in options],
                    "answers": answers,
                    "explanation": explanation.strip() if explanation else ""
                })
    
    return quizzes

def extract_from_markdown(text):
    """Extract quizzes from markdown section."""
    quizzes = []
    
    # Find quiz section
    section_match = QUIZ_SECTION_RE.search(text)
    if not section_match:
        return []
    
    section_content = section_match.group(1)
    
    # Find all quiz blocks in the section
    for match in QUIZ_BLOCK_RE.finditer(section_content):
        question = match.group(1).strip()
        options_text = match.group(2).strip()
        answers_text = match.group(3).strip()
        explanation = match.group(4).strip()
        
        # Parse options (format: - A) Option text)
        options = []
        option_lines = options_text.split('\n')
        for line in option_lines:
            line = line.strip()
            if line:
                # Remove leading "- A)" or similar
                option_match = re.match(r'-\s*[A-Z]\)\s*(.+)', line)
                if option_match:
                    options.append(option_match.group(1).strip())
        
        # Parse answers (format: "A, C" or "A" or "0, 2")
        answers = []
        answer_parts = [a.strip() for a in answers_text.split(',')]
        for part in answer_parts:
            # Try to convert letter to index (A=0, B=1, etc.)
            if part.isalpha() and len(part) == 1:
                answers.append(ord(part.upper()) - ord('A'))
            elif part.isdigit():
                answers.append(int(part))
        
        if question and options and answers:
            quizzes.append({
                "q": question,
                "options": options,
                "answers": answers,
                "explanation": explanation
            })
    
    return quizzes

# Collect all quizzes
quizzes = []
categories = ["algorithm", "system-design", "devops", "leetcode", "agentic", "machine-learning"]

for cat in categories:
    folder = NOTES / cat
    if not folder.exists():
        continue
    
    for md_file in folder.rglob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        source = str(md_file.relative_to(ROOT))
        
        found = []
        found += extract_from_frontmatter(text)
        found += extract_from_markdown(text)
        
        for quiz in found:
            quiz["category"] = cat
            quiz["source"] = source
            quizzes.append(quiz)

# Deduplicate based on question and options
seen = set()
unique = []
for q in quizzes:
    key = (q['q'], tuple(q['options']))
    if key in seen:
        continue
    seen.add(key)
    unique.append(q)

# Write output
OUT_FILE = OUT / "quizzes.json"
OUT_FILE.write_text(json.dumps(unique, ensure_ascii=False, indent=2))
print(f"Wrote {len(unique)} quizzes to {OUT_FILE}")


