import { useState, useEffect } from 'react';
import type { Flashcard, Quiz, Note } from '@/types';

export function useData() {
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [quizzes, setQuizzes] = useState<Quiz[]>([]);
  const [notes, setNotes] = useState<Note[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch('./artifacts/flashcards.json').then(res => res.json()),
      fetch('./artifacts/quizzes.json').then(res => res.json()),
      fetch('./artifacts/notes.json').then(res => res.json()).catch(() => []) // Handle missing notes.json gracefully
    ]).then(([fcData, qData, notesData]) => {
      setFlashcards(fcData);
      setQuizzes(qData);
      setNotes(notesData);
      setLoading(false);
    }).catch(err => {
      console.error("Failed to load data", err);
      setLoading(false);
    });
  }, []);

  return { flashcards, quizzes, notes, loading };
}
