import React, { useState } from 'react';
import type { Note } from '@/types';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { CategoryBadge } from './CategoryBadge';
import { NoteView } from './NoteView';

interface NotesListProps {
  notes: Note[];
}

export const NotesList: React.FC<NotesListProps> = ({ notes }) => {
  const [selectedNote, setSelectedNote] = useState<Note | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const categories = ['all', ...Array.from(new Set(notes.map(n => n.category))).sort()];
  
  const filteredNotes = notes.filter(n => 
    selectedCategory === 'all' || n.category === selectedCategory
  );

  if (selectedNote) {
    return (
      <div className="space-y-4">
        <button 
          onClick={() => setSelectedNote(null)}
          className="text-sm text-muted-foreground hover:underline mb-2 flex items-center gap-1"
        >
          ← Back to List
        </button>
        <NoteView note={selectedNote} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2">
        {categories.map(cat => (
          <Badge 
            key={cat}
            variant={selectedCategory === cat ? "default" : "outline"}
            className="cursor-pointer capitalize"
            onClick={() => setSelectedCategory(cat)}
          >
            {cat}
          </Badge>
        ))}
      </div>

      <ScrollArea className="h-[calc(100vh-250px)]">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredNotes.map((note) => (
            <Card 
              key={note.path} 
              className="cursor-pointer hover:bg-muted/50 transition-colors"
              onClick={() => setSelectedNote(note)}
            >
              <CardHeader>
                <div className="flex justify-between items-start gap-2">
                  <CardTitle className="text-lg leading-tight">{note.title}</CardTitle>
                  <CategoryBadge category={note.category} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1 mt-2">
                  {note.tags?.map(tag => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      #{tag}
                    </Badge>
                  ))}
                </div>
                <div className="text-xs text-muted-foreground mt-4">
                  {note.date}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};

