import { useState, useEffect } from 'react';
import type { Quiz } from '@/types';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { CategoryBadge } from '@/components/CategoryBadge';
import { ChevronLeft, ChevronRight, Shuffle, CheckCircle2, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface QuizDeckProps {
  quizzes: Quiz[];
}

export function QuizDeck({ quizzes }: QuizDeckProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedOptions, setSelectedOptions] = useState<Set<number>>(new Set());
  const [submitted, setSubmitted] = useState(false);

  useEffect(() => {
    setCurrentIndex(0);
    resetState();
  }, [quizzes]);

  const resetState = () => {
    setSelectedOptions(new Set());
    setSubmitted(false);
  };

  if (quizzes.length === 0) {
    return (
      <Card className="w-full max-w-2xl mx-auto mt-8 min-h-[300px] flex items-center justify-center text-muted-foreground">
        No quizzes match your filter.
      </Card>
    );
  }

  const currentQuiz = quizzes[currentIndex];

  const toggleOption = (index: number) => {
    if (submitted) return;
    const newSelected = new Set(selectedOptions);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedOptions(newSelected);
  };

  const submitQuiz = () => {
    setSubmitted(true);
  };

  const nextQuiz = () => {
    setCurrentIndex((prev) => (prev + 1) % quizzes.length);
    resetState();
  };

  const prevQuiz = () => {
    setCurrentIndex((prev) => (prev - 1 + quizzes.length) % quizzes.length);
    resetState();
  };

  const randomQuiz = () => {
    const next = Math.floor(Math.random() * quizzes.length);
    setCurrentIndex(next);
    resetState();
  };

  // Validation Logic
  const correctAnswers = new Set(currentQuiz.answers);
  const isCorrect = submitted && 
    correctAnswers.size === selectedOptions.size && 
    [...correctAnswers].every(a => selectedOptions.has(a));
  
  const isPartial = submitted && !isCorrect && 
    [...selectedOptions].some(a => correctAnswers.has(a));

  return (
    <div className="flex flex-col items-center space-y-6 w-full max-w-2xl mx-auto">
      <Card className="w-full">
        <CardHeader className="pb-2">
            <div className="flex gap-2 mb-2">
                <CategoryBadge category={currentQuiz.category} />
                <span className="text-xs text-muted-foreground bg-secondary px-2 py-1 rounded-md font-medium flex items-center">
                    {currentQuiz.source}
                </span>
            </div>
        </CardHeader>
        <CardContent className="space-y-6 pt-4">
          <div className="text-xl font-medium leading-relaxed">
            {currentQuiz.q}
          </div>

          <div className="grid gap-3">
            {currentQuiz.options.map((option, idx) => {
              const isSelected = selectedOptions.has(idx);
              const isAnswer = correctAnswers.has(idx);
              
              let variantClass = "border-border hover:bg-secondary/50";
              
              if (submitted) {
                if (isAnswer) variantClass = "border-green-500 bg-green-50 dark:bg-green-950/20";
                else if (isSelected && !isAnswer) variantClass = "border-red-500 bg-red-50 dark:bg-red-950/20";
                else variantClass = "opacity-60";
              } else if (isSelected) {
                variantClass = "border-primary bg-primary/5 ring-1 ring-primary";
              }

              return (
                <div 
                  key={idx}
                  onClick={() => toggleOption(idx)}
                  className={cn(
                    "p-4 rounded-lg border-2 cursor-pointer transition-all flex items-start gap-3",
                    variantClass
                  )}
                >
                    <div className={cn(
                        "mt-1 h-4 w-4 rounded border flex items-center justify-center shrink-0",
                        isSelected ? "bg-primary border-primary text-primary-foreground" : "border-muted-foreground"
                    )}>
                        {isSelected && <div className="h-2 w-2 rounded-full bg-current" />}
                    </div>
                    <span className="text-sm leading-relaxed">{option}</span>
                    
                    {submitted && isAnswer && <CheckCircle2 className="h-5 w-5 text-green-600 ml-auto shrink-0" />}
                    {submitted && isSelected && !isAnswer && <XCircle className="h-5 w-5 text-red-600 ml-auto shrink-0" />}
                </div>
              );
            })}
          </div>

          {submitted && (
            <div className="animate-in fade-in slide-in-from-top-2 duration-300 space-y-4">
                <div className={cn(
                    "p-4 rounded-md text-center font-medium border",
                    isCorrect ? "bg-green-50 text-green-800 border-green-200 dark:bg-green-950/30 dark:text-green-400 dark:border-green-900" :
                    isPartial ? "bg-orange-50 text-orange-800 border-orange-200 dark:bg-orange-950/30 dark:text-orange-400 dark:border-orange-900" :
                    "bg-red-50 text-red-800 border-red-200 dark:bg-red-950/30 dark:text-red-400 dark:border-red-900"
                )}>
                    {isCorrect ? "Correct! Well done!" : 
                     isPartial ? "Partially correct." : 
                     "Incorrect. Review the explanation below."}
                </div>
                
                <div className="bg-muted/50 p-4 rounded-md border-l-4 border-primary">
                    <div className="font-semibold mb-1">Explanation</div>
                    <div className="text-sm text-muted-foreground leading-relaxed">
                        {currentQuiz.explanation || "No explanation available."}
                    </div>
                </div>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex flex-col w-full gap-4 max-w-md">
        {!submitted ? (
            <Button size="lg" onClick={submitQuiz} disabled={selectedOptions.size === 0}>
                Submit Answer
            </Button>
        ) : (
            <div className="flex items-center justify-between gap-4">
                <Button variant="outline" onClick={prevQuiz} className="flex-1">
                    <ChevronLeft className="mr-2 h-4 w-4" /> Prev
                </Button>
                <Button variant="secondary" onClick={randomQuiz} className="flex-1">
                    <Shuffle className="mr-2 h-4 w-4" /> Random
                </Button>
                <Button variant="default" onClick={nextQuiz} className="flex-1">
                    Next <ChevronRight className="ml-2 h-4 w-4" />
                </Button>
            </div>
        )}
        
        <div className="text-center text-xs text-muted-foreground">
            Quiz {currentIndex + 1} of {quizzes.length}
        </div>
      </div>
    </div>
  );
}

