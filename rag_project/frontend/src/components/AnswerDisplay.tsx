import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { CheckCircle, AlertCircle, Info, Copy, ThumbsUp, ThumbsDown } from 'lucide-react';
import { toast } from 'sonner@2.0.3';

interface AnswerDisplayProps {
  currentAnswer: {
    question: string;
    answer: string;
    confidence: number;
    relevantSections: string[];
    sources: string[];
  } | null;
  onHighlightSections: (sections: string[]) => void;
}

export function AnswerDisplay({ currentAnswer, onHighlightSections }: AnswerDisplayProps) {
  if (!currentAnswer) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5" />
            Answer
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <Info className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>Ask a question to see the AI analysis.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getConfidenceIcon = (confidence: number) => {
    if (confidence > 0.8) return <CheckCircle className="h-4 w-4 text-green-600" />;
    if (confidence > 0.6) return <AlertCircle className="h-4 w-4 text-yellow-600" />;
    return <AlertCircle className="h-4 w-4 text-red-600" />;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return "default";
    if (confidence > 0.6) return "secondary";
    return "destructive";
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(currentAnswer.answer);
    toast("Answer copied to clipboard");
  };

  const handleFeedback = (positive: boolean) => {
    toast(positive ? "Thank you for your feedback!" : "Feedback recorded. We'll improve our responses.");
  };

  return (
    <Card className="shadow-lg border-2 hover:shadow-xl transition-all duration-300" style={{ borderColor: 'var(--border)' }}>
      <CardHeader className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-950/20 dark:to-cyan-950/20">
        <CardTitle className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-gradient-to-r from-teal-500 to-cyan-500 text-white">
            <CheckCircle className="h-5 w-5" />
          </div>
          <span className="bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text text-transparent font-bold">
            Answer
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge className="bg-gradient-to-r from-blue-500 to-purple-500 text-white border-0">Question</Badge>
          </div>
          <p className="font-semibold text-lg bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">{currentAnswer.question}</p>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Badge className="bg-gradient-to-r from-teal-500 to-cyan-500 text-white border-0">Answer</Badge>
              <Badge variant={getConfidenceColor(currentAnswer.confidence)} className="flex items-center gap-1 bg-gradient-to-r from-green-500 to-emerald-500 text-white border-0">
                {getConfidenceIcon(currentAnswer.confidence)}
                {Math.round(currentAnswer.confidence * 100)}% confident
              </Badge>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={copyToClipboard}
                className="h-8 w-8 p-0"
              >
                <Copy className="h-3 w-3" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleFeedback(true)}
                className="h-8 w-8 p-0"
              >
                <ThumbsUp className="h-3 w-3" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleFeedback(false)}
                className="h-8 w-8 p-0"
              >
                <ThumbsDown className="h-3 w-3" />
              </Button>
            </div>
          </div>
          <div className="bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-teal-950/30 dark:to-cyan-950/30 p-6 rounded-xl border-2 border-teal-200 dark:border-teal-700 shadow-inner">
            <p className="whitespace-pre-wrap text-gray-800 dark:text-gray-200 leading-relaxed">{currentAnswer.answer}</p>
          </div>
        </div>

        {currentAnswer.relevantSections.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <p className="font-medium">Relevant sections found:</p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => onHighlightSections(currentAnswer.relevantSections)}
              >
                Highlight in document
              </Button>
            </div>
            <div className="flex flex-wrap gap-2">
              {currentAnswer.relevantSections.map((section, index) => (
                <Badge key={index} variant="outline">
                  {section}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {currentAnswer.sources.length > 0 && (
          <div className="space-y-2">
            <p className="font-medium">Sources:</p>
            <div className="space-y-1">
              {currentAnswer.sources.map((source, index) => (
                <p key={index} className="text-sm text-muted-foreground">
                  • {source}
                </p>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}