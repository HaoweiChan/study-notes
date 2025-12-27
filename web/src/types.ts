export interface Flashcard {
  q: string;
  a: string;
  category: string;
  source: string;
}

export interface Quiz {
  q: string;
  options: string[];
  answers: number[];
  explanation: string;
  category: string;
  source: string;
}

export interface Note {
  title: string;
  category: string;
  slug: string;
  path: string;
  tags: string[];
  date: string;
}

export const CATEGORY_COLORS: Record<string, string> = {
  'algorithm': 'bg-blue-600',
  'devops': 'bg-green-600',
  'machine-learning': 'bg-purple-600',
  'system-design': 'bg-red-600',
  'leetcode': 'bg-orange-600',
  'agentic': 'bg-cyan-600',
};

export const FALLBACK_COLORS = [
  'bg-pink-600',
  'bg-amber-800',
  'bg-indigo-600',
  'bg-lime-600',
  'bg-teal-600',
];
