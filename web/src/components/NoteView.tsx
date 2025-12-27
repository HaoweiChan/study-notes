import React, { useEffect, useState } from 'react';
import type { Note } from '@/types';
import ReactMarkdown from 'react-markdown';
import { Card, CardContent } from '@/components/ui/card';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface NoteViewProps {
  note: Note;
}

export const NoteView: React.FC<NoteViewProps> = ({ note }) => {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(note.path)
      .then(res => res.text())
      .then(text => {
        // Remove frontmatter
        const contentWithoutFrontmatter = text.replace(/^---[\s\S]*?---\n/, '');
        setContent(contentWithoutFrontmatter);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load note content", err);
        setContent("Failed to load content.");
        setLoading(false);
      });
  }, [note.path]);

  if (loading) {
    return <div className="p-8 text-center">Loading note...</div>;
  }

  return (
    <Card>
      <CardContent className="pt-6">
        <article className="prose prose-slate dark:prose-invert max-w-none">
          <ReactMarkdown
            components={{
              code({node, inline, className, children, ...props}: any) {
                const match = /language-(\w+)/.exec(className || '')
                return !inline && match ? (
                  <SyntaxHighlighter
                    {...props}
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                )
              }
            }}
          >
            {content}
          </ReactMarkdown>
        </article>
      </CardContent>
    </Card>
  );
};
