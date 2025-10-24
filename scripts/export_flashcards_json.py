#!/usr/bin/env python3
# scripts/export_flashcards_json.py
import re, json, sys
from pathlib import Path
import frontmatter

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"
OUT = ROOT / "docs" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

FENCE_RE = re.compile(r"```flashcard\s*\n(.*?)\n```", re.S|re.I)
FLASH_SECTION_RE = re.compile(r"^##+\s*Flashcards\s*(.*?)^(?:##\s|\Z)", re.S|re.I|re.M)
LIST_QA_RE = re.compile(r"^\s*[-*]\s*(.+?)\s*(?::::|\|\|\||::)\s*(.+)\s*$")

def extract_from_frontmatter(text):
    try:
        fm = frontmatter.loads(text)
    except Exception:
        return []
    cards = []
    if isinstance(fm.metadata.get("flashcards"), list):
        for item in fm.metadata["flashcards"]:
            q = item.get("q") or item.get("Q") or item.get("question")
            a = item.get("a") or item.get("A") or item.get("answer")
            if q and a:
                cards.append((q.strip(), a.strip()))
    return cards

def extract_fenced(text):
    out=[]
    for m in FENCE_RE.finditer(text):
        block = m.group(1).strip()
        parts = re.split(r"\n-{3,}\n", block)
        if len(parts)==2:
            q = re.sub(r"^Q:\s*","",parts[0],flags=re.I).strip()
            a = re.sub(r"^A:\s*","",parts[1],flags=re.I).strip()
            out.append((q,a))
        else:
            lines = block.splitlines()
            if len(lines)>=2:
                q = lines[0].strip(); a = "\n".join(lines[1:]).strip()
                out.append((q,a))
    return out

def extract_section(text):
    out=[]
    for m in FLASH_SECTION_RE.finditer(text):
        section = m.group(1)
        for line in section.splitlines():
            lm = LIST_QA_RE.match(line.strip())
            if lm:
                out.append((lm.group(1).strip(), lm.group(2).strip()))
    return out

cards = []
for cat in ("algorithm", "system-design", "devops", "leetcode", "agentic", "machine-learning"):
    folder = NOTES / cat
    if not folder.exists(): continue
    for md in folder.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        source = str(md.relative_to(ROOT))
        found = []
        found += extract_from_frontmatter(text)
        found += extract_fenced(text)
        found += extract_section(text)
        for q,a in found:
            cards.append({
                "q": q,
                "a": a,
                "category": cat,
                "source": source
            })

# dedupe (q,a)
seen=set()
unique=[]
for c in cards:
    key=(c['q'],c['a'])
    if key in seen: continue
    seen.add(key); unique.append(c)

OUT_FILE = OUT / "flashcards.json"
OUT_FILE.write_text(json.dumps(unique, ensure_ascii=False, indent=2))
print("Wrote", OUT_FILE)
