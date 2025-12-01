import { useState, useEffect } from 'react';
import type { Flashcard, Quiz } from '@/types';

export function useData() {
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [quizzes, setQuizzes] = useState<Quiz[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch('./artifacts/flashcards.json').then(res => res.json()),
      fetch('./artifacts/quizzes.json').then(res => res.json())
    ]).then(([fcData, qData]) => {
      setFlashcards(fcData);
      setQuizzes(qData);
      setLoading(false);
    }).catch(err => {
      console.error("Failed to load data", err);
      setLoading(false);
    });
  }, []);

  return { flashcards, quizzes, loading };
}

