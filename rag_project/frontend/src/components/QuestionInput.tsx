import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Send, MessageSquare } from 'lucide-react';

interface QuestionInputProps {
  onSubmitQuestion: (question: string) => void;
  isLoading: boolean;
  disabled: boolean;
}

export function QuestionInput({ onSubmitQuestion, isLoading, disabled }: QuestionInputProps) {
  const [question, setQuestion] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (question.trim() && !isLoading) {
      onSubmitQuestion(question.trim());
      setQuestion('');
    }
  };

  const suggestedQuestions = [
    "What is the main purpose of this document?",
    "Can you summarize the key points?",
    "What are the technical requirements mentioned?",
    "Who is the target audience for this document?",
    "What are the next steps outlined?"
  ];

  return (
    <Card className="shadow-lg border-2 hover:shadow-xl transition-all duration-300" style={{ borderColor: 'var(--border)' }}>
      <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-950/20 dark:to-pink-950/20">
        <CardTitle className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 text-white">
            <MessageSquare className="h-5 w-5" />
          </div>
          <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent font-bold">
            Ask a Question
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          <Textarea
            placeholder="Type your question about the document here..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={disabled}
            rows={3}
            className="resize-none border-2 focus:border-purple-400 focus:ring-purple-400 transition-all duration-300"
          />
          <Button 
            type="submit" 
            disabled={!question.trim() || isLoading || disabled}
            className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
                Analyzing...
              </>
            ) : (
              <>
                <Send className="h-4 w-4 mr-2" />
                Ask Question
              </>
            )}
          </Button>
        </form>

        {!disabled && (
          <div className="space-y-3">
            <p className="text-sm font-semibold text-gray-700 dark:text-gray-300">💡 Suggested questions:</p>
            <div className="space-y-2">
              {suggestedQuestions.map((suggested, index) => (
                <button
                  key={index}
                  onClick={() => setQuestion(suggested)}
                  className="text-sm text-gray-600 dark:text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 text-left block w-full p-3 rounded-lg hover:bg-gradient-to-r hover:from-purple-50 hover:to-pink-50 dark:hover:from-purple-950/30 dark:hover:to-pink-950/30 transition-all duration-300 border border-transparent hover:border-purple-200 dark:hover:border-purple-700"
                  disabled={isLoading}
                >
                  {suggested}
                </button>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}