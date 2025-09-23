import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import { FileText, Eye } from 'lucide-react';

interface DocumentViewerProps {
  document: { file: File; content: string } | null;
  highlightedSections?: string[];
}

export function DocumentViewer({ document, highlightedSections = [] }: DocumentViewerProps) {
  if (!document) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Document Viewer
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12 text-muted-foreground">
            <FileText className="h-16 w-16 mx-auto mb-4 opacity-50" />
            <h3 className="mb-2">No document uploaded</h3>
            <p className="text-sm">Upload a document to view its content here.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const highlightContent = (content: string, sections: string[]) => {
    if (sections.length === 0) return content;
    
    let highlightedContent = content;
    sections.forEach((section) => {
      const regex = new RegExp(`(${section})`, 'gi');
      highlightedContent = highlightedContent.replace(
        regex,
        '<mark class="bg-yellow-200 dark:bg-yellow-800 px-1 rounded">$1</mark>'
      );
    });
    
    return highlightedContent;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Eye className="h-5 w-5" />
          Document Viewer
        </CardTitle>
        <div className="flex items-center gap-2 mt-2">
          <Badge variant="outline">{document.file.name}</Badge>
          <Badge variant="secondary">
            {(document.file.size / 1024).toFixed(1)} KB
          </Badge>
          {highlightedSections.length > 0 && (
            <Badge variant="default">
              {highlightedSections.length} sections highlighted
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96">
          <div 
            className="prose prose-sm max-w-none dark:prose-invert whitespace-pre-wrap font-mono text-sm leading-relaxed"
            dangerouslySetInnerHTML={{ 
              __html: highlightContent(document.content, highlightedSections) 
            }}
          />
        </ScrollArea>
      </CardContent>
    </Card>
  );
}