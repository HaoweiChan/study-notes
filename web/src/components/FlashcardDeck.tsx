import { useState, useEffect } from 'react';
import type { Flashcard } from '@/types';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { CategoryBadge } from '@/components/CategoryBadge';
import { ChevronLeft, ChevronRight, Shuffle, Star } from 'lucide-react';
import { usePersistedState } from '@/hooks/use-persisted-state';

interface FlashcardDeckProps {
  cards: Flashcard[];
  allCards: Flashcard[]; // Needed for stable bookmarking by global index/content
}

export function FlashcardDeck({ cards, allCards }: FlashcardDeckProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [bookmarks, setBookmarks] = usePersistedState<number[]>('bookmarked_cards', []);

  // Reset index when deck changes (filtered)
  useEffect(() => {
    setCurrentIndex(0);
    setShowAnswer(false);
  }, [cards]);

  if (cards.length === 0) {
    return (
      <Card className="w-full max-w-2xl mx-auto mt-8 min-h-[300px] flex items-center justify-center text-muted-foreground">
        No cards match your filter.
      </Card>
    );
  }

  const currentCard = cards[currentIndex];
  // Find original index to track bookmarks reliably
  // Assuming cards are unique by q+a or we use the index from allCards if possible.
  // A better way is to generate a hash or ID. For now, we'll use the index in allCards.
  const originalIndex = allCards.findIndex(c => c.q === currentCard.q && c.category === currentCard.category);
  const isBookmarked = bookmarks.includes(originalIndex);

  const toggleBookmark = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isBookmarked) {
      setBookmarks(bookmarks.filter(id => id !== originalIndex));
    } else {
      setBookmarks([...bookmarks, originalIndex]);
    }
  };

  const nextCard = () => {
    setCurrentIndex((prev) => (prev + 1) % cards.length);
    setShowAnswer(false);
  };

  const prevCard = () => {
    setCurrentIndex((prev) => (prev - 1 + cards.length) % cards.length);
    setShowAnswer(false);
  };

  const randomCard = () => {
    const next = Math.floor(Math.random() * cards.length);
    setCurrentIndex(next);
    setShowAnswer(false);
  };

  return (
    <div className="flex flex-col items-center space-y-6 w-full max-w-2xl mx-auto">
      <Card 
        className="w-full min-h-[400px] cursor-pointer flex flex-col relative transition-all hover:shadow-md"
        onClick={() => setShowAnswer(!showAnswer)}
      >
        <div className="absolute top-4 left-4 z-10">
            <Button 
                variant="ghost" 
                size="icon" 
                className={isBookmarked ? "text-yellow-400 hover:text-yellow-500" : "text-muted-foreground hover:text-yellow-400"}
                onClick={toggleBookmark}
            >
                <Star className={isBookmarked ? "fill-current" : ""} />
            </Button>
        </div>

        <CardContent className="flex-1 flex flex-col justify-center items-center p-8 text-center space-y-8">
          <div className="text-2xl font-medium leading-relaxed">
            {currentCard.q}
          </div>
          
          {showAnswer && (
            <div className="text-lg text-muted-foreground animate-in fade-in slide-in-from-bottom-2 duration-300 border-t pt-6 w-full whitespace-pre-wrap">
              {currentCard.a}
            </div>
          )}
        </CardContent>

        <CardFooter className="border-t p-4 flex justify-center gap-2 bg-muted/20 w-full">
            <CategoryBadge category={currentCard.category} />
            <span className="text-xs text-muted-foreground bg-secondary px-2 py-1 rounded-md font-medium">
                {currentCard.source}
            </span>
        </CardFooter>
      </Card>

      <div className="flex flex-col w-full gap-4 max-w-md">
        <Button 
            size="lg" 
            onClick={() => setShowAnswer(true)}
            className={showAnswer ? "invisible" : ""}
        >
            Show Answer
        </Button>

        <div className="flex items-center justify-between gap-4">
            <Button variant="outline" onClick={prevCard} className="flex-1">
                <ChevronLeft className="mr-2 h-4 w-4" /> Prev
            </Button>
            <Button variant="secondary" onClick={randomCard} className="flex-1">
                <Shuffle className="mr-2 h-4 w-4" /> Random
            </Button>
            <Button variant="outline" onClick={nextCard} className="flex-1">
                Next <ChevronRight className="ml-2 h-4 w-4" />
            </Button>
        </div>
        
        <div className="text-center text-xs text-muted-foreground">
            Card {currentIndex + 1} of {cards.length}
        </div>
      </div>
    </div>
  );
}

