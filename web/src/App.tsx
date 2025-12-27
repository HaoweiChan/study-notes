import { useState, useEffect } from 'react';
import { useData } from '@/hooks/use-data';
import { FlashcardDeck } from '@/components/FlashcardDeck';
import { QuizDeck } from '@/components/QuizDeck';
import { NotesList } from '@/components/NotesList';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Moon, Sun, Search, BookOpen, GraduationCap, Star, FileText } from 'lucide-react';
import { usePersistedState } from '@/hooks/use-persisted-state';

function App() {
  const { flashcards, quizzes, notes, loading } = useData();
  const [category, setCategory] = usePersistedState('selected_category', 'all');
  const [search, setSearch] = useState('');
  const [isDark, setIsDark] = usePersistedState('dark_mode', false);
  const [bookmarkedOnly, setBookmarkedOnly] = useState(false);
  const [bookmarks] = usePersistedState<number[]>('bookmarked_cards', []);
  const [activeTab, setActiveTab] = useState('flashcards');

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
    ...quizzes.map(q => q.category),
    ...notes.map(n => n.category)
  ])).sort();

  const filteredFlashcards = flashcards.filter(c => {
    const matchCategory = category === 'all' || c.category === category;
    const matchSearch = search === '' || 
      c.q.toLowerCase().includes(search.toLowerCase()) || 
      c.a.toLowerCase().includes(search.toLowerCase());
    
    // Bookmark filter requires finding the original index
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

  const filteredNotes = notes.filter(n => {
    const matchCategory = category === 'all' || n.category === category;
    const matchSearch = search === '' || 
      n.title.toLowerCase().includes(search.toLowerCase()) || 
      n.tags?.some(t => t.toLowerCase().includes(search.toLowerCase()));
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
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full space-y-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <TabsList className="grid w-full sm:w-[600px] grid-cols-3">
                <TabsTrigger value="flashcards" className="gap-2">
                    <BookOpen className="h-4 w-4" /> Flashcards
                </TabsTrigger>
                <TabsTrigger value="quizzes" className="gap-2">
                    <GraduationCap className="h-4 w-4" /> Quizzes
                </TabsTrigger>
                <TabsTrigger value="notes" className="gap-2">
                    <FileText className="h-4 w-4" /> Notes
                </TabsTrigger>
            </TabsList>

            <div className="flex items-center gap-2">
                {/* Show Bookmarked Only button if active tab is flashcards */}
                {activeTab === 'flashcards' && (
                     <Button 
                        variant={bookmarkedOnly ? "secondary" : "outline"} 
                        size="sm" 
                        onClick={() => setBookmarkedOnly(!bookmarkedOnly)}
                        className="gap-2 h-9"
                    >
                        <Star className={bookmarkedOnly ? "fill-yellow-400 text-yellow-400" : "text-muted-foreground"} size={16} />
                        <span className="hidden sm:inline">{bookmarkedOnly ? "Show All" : "Bookmarked Only"}</span>
                        <span className="sm:hidden">{bookmarkedOnly ? "All" : "Stars"}</span>
                    </Button>
                )}
            </div>
          </div>

          <TabsContent value="flashcards" className="focus-visible:outline-none focus-visible:ring-0 mt-0">
            <FlashcardDeck cards={filteredFlashcards} allCards={flashcards} />
          </TabsContent>

          <TabsContent value="quizzes" className="focus-visible:outline-none focus-visible:ring-0 mt-0">
            <QuizDeck quizzes={filteredQuizzes} />
          </TabsContent>

          <TabsContent value="notes" className="focus-visible:outline-none focus-visible:ring-0 mt-0">
            <NotesList notes={filteredNotes} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

export default App;
