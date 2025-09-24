import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { Clock, MessageSquare, Lightbulb } from 'lucide-react';

interface QAItem {
  id: string;
  question: string;
  answer: string;
  confidence: number;
  timestamp: Date;
  relevantSections: string[];
}

interface AnalysisHistoryProps {
  qaHistory: QAItem[];
}

export function AnalysisHistory({ qaHistory }: AnalysisHistoryProps) {
  if (qaHistory.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Analysis History
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No questions asked yet.</p>
            <p className="text-sm">Upload a document and ask questions to see the analysis history.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Clock className="h-5 w-5" />
          Analysis History ({qaHistory.length})
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96">
          <div className="space-y-4">
            {qaHistory.map((item) => (
              <div key={item.id} className="border rounded-lg p-4 space-y-3">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge variant="outline" className="text-xs">
                        Question
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {item.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="font-medium text-sm">{item.question}</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">
                      Answer
                    </Badge>
                    <Badge 
                      variant={item.confidence > 0.8 ? "default" : item.confidence > 0.6 ? "secondary" : "outline"}
                      className="text-xs"
                    >
                      {Math.round(item.confidence * 100)}% confidence
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{item.answer}</p>
                </div>

                {item.relevantSections.length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Lightbulb className="h-3 w-3" />
                      <span className="text-xs font-medium">Relevant sections:</span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {item.relevantSections.map((section, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {section}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}