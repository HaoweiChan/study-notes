import { useState, useEffect } from 'react';
import { useData } from '@/hooks/use-data';
import { FlashcardDeck } from '@/components/FlashcardDeck';
import { QuizDeck } from '@/components/QuizDeck';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Moon, Sun, Search, BookOpen, GraduationCap, Star } from 'lucide-react';
import { usePersistedState } from '@/hooks/use-persisted-state';

function App() {
  const { flashcards, quizzes, loading } = useData();
  const [category, setCategory] = usePersistedState('selected_category', 'all');
  const [search, setSearch] = useState('');
  const [isDark, setIsDark] = usePersistedState('dark_mode', false);
  const [bookmarkedOnly, setBookmarkedOnly] = useState(false);
  const [bookmarks] = usePersistedState<number[]>('bookmarked_cards', []);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  if (loading) {
    return <div className="flex items-center justify-center h-screen">Loading...</div>;
  }

  const categories = Array.from(new Set([
    ...flashcards.map(c => c.category),
    ...quizzes.map(q => q.category)
  ])).sort();

  const filteredFlashcards = flashcards.filter(c => {
    const matchCategory = category === 'all' || c.category === category;
    const matchSearch = search === '' || 
      c.q.toLowerCase().includes(search.toLowerCase()) || 
      c.a.toLowerCase().includes(search.toLowerCase());
    
    // Bookmark filter requires finding the original index
    // This is a bit expensive O(N^2) effectively if implemented naively in filter
    // Optimization: Pre-compute original indices or assume unique identity
    const originalIdx = flashcards.indexOf(c); // Determine index in original array
    const matchBookmark = !bookmarkedOnly || bookmarks.includes(originalIdx);

    return matchCategory && matchSearch && matchBookmark;
  });

  const filteredQuizzes = quizzes.filter(q => {
    const matchCategory = category === 'all' || q.category === category;
    const matchSearch = search === '' || 
      q.q.toLowerCase().includes(search.toLowerCase());
    return matchCategory && matchSearch;
  });

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
      <header className="border-b sticky top-0 z-50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 font-bold text-xl hidden md:flex">
            <BookOpen className="h-6 w-6 text-primary" />
            <span>Study Notes</span>
          </div>

          <div className="flex items-center gap-4 flex-1 max-w-2xl justify-center">
            <div className="relative w-full max-w-xs sm:max-w-sm">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                    type="search"
                    placeholder="Search..."
                    className="pl-9 w-full bg-muted/50"
                    value={search}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
                />
            </div>
            
            <Select value={category} onValueChange={setCategory}>
                <SelectTrigger className="w-[140px] hidden sm:flex">
                    <SelectValue placeholder="Category" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="all">All Categories</SelectItem>
                    {categories.map(c => (
                        <SelectItem key={c} value={c}>{c}</SelectItem>
                    ))}
                </SelectContent>
            </Select>
          </div>

          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setIsDark(!isDark)}
            title="Toggle Theme"
          >
            {isDark ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Tabs defaultValue="flashcards" className="w-full space-y-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <TabsList className="grid w-full sm:w-[400px] grid-cols-2">
                <TabsTrigger value="flashcards" className="gap-2">
                    <BookOpen className="h-4 w-4" /> Flashcards
                </TabsTrigger>
                <TabsTrigger value="quizzes" className="gap-2">
                    <GraduationCap className="h-4 w-4" /> Quizzes
                </TabsTrigger>
            </TabsList>

            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                {category !== 'all' && (
                    <span className="bg-primary/10 text-primary px-2 py-1 rounded-md text-xs font-medium uppercase">
                        {category}
                    </span>
                )}
                {bookmarkedOnly && (
                    <span className="bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 px-2 py-1 rounded-md text-xs font-medium uppercase">
                        Bookmarked
                    </span>
                )}
            </div>
          </div>

          <TabsContent value="flashcards" className="space-y-4 focus-visible:outline-none focus-visible:ring-0">
            <div className="flex justify-end mb-4">
                 <Button 
                    variant={bookmarkedOnly ? "secondary" : "outline"} 
                    size="sm" 
                    onClick={() => setBookmarkedOnly(!bookmarkedOnly)}
                    className="gap-2"
                >
                    <Star className={bookmarkedOnly ? "fill-yellow-400 text-yellow-400" : "text-muted-foreground"} size={16} />
                    {bookmarkedOnly ? "Show All" : "Bookmarked Only"}
                </Button>
            </div>
            <FlashcardDeck cards={filteredFlashcards} allCards={flashcards} />
          </TabsContent>

          <TabsContent value="quizzes" className="focus-visible:outline-none focus-visible:ring-0">
            <QuizDeck quizzes={filteredQuizzes} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

export default App;
