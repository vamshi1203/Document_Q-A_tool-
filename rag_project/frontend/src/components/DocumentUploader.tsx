import { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Upload, File, X } from 'lucide-react';

interface DocumentUploaderProps {
  onDocumentUpload: (file: File, content: string) => void;
  uploadedDocument: { file: File; content: string } | null;
  onRemoveDocument: () => void;
}

export function DocumentUploader({ onDocumentUpload, uploadedDocument, onRemoveDocument }: DocumentUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    const file = files[0];
    
    if (file && (file.type === 'text/plain' || file.type === 'application/pdf' || file.name.endsWith('.txt'))) {
      processFile(file);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  }, []);

  const processFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      // For demo purposes, we'll simulate content extraction
      const mockContent = `Sample document content from ${file.name}\n\nThis is a sample document that contains information about various topics. The document includes details about business processes, technical specifications, and operational procedures.\n\nKey sections include:\n- Introduction and Overview\n- Technical Requirements\n- Implementation Guidelines\n- Best Practices\n- Conclusion and Next Steps\n\nThis content would normally be extracted from the actual uploaded file using appropriate parsing libraries.`;
      
      onDocumentUpload(file, mockContent);
    };
    reader.readAsText(file);
  };

  if (uploadedDocument) {
    return (
      <Card className="shadow-lg border-2 bg-gradient-to-br from-green-50 to-teal-50 dark:from-green-950/20 dark:to-teal-950/20" style={{ borderColor: 'var(--border)' }}>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <div className="p-2 rounded-lg bg-gradient-to-r from-green-500 to-teal-500 text-white">
                <File className="h-5 w-5" />
              </div>
              <span className="bg-gradient-to-r from-green-600 to-teal-600 bg-clip-text text-transparent font-bold">
                Uploaded Document
              </span>
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={onRemoveDocument}
              className="h-8 w-8 p-0 hover:bg-red-50 hover:border-red-300 hover:text-red-600 transition-all duration-300"
            >
              <X className="h-4 w-4" />
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <p className="font-semibold text-lg">{uploadedDocument.file.name}</p>
            <div className="inline-block px-3 py-1 bg-gradient-to-r from-green-100 to-teal-100 dark:from-green-900/30 dark:to-teal-900/30 rounded-full">
              <p className="text-sm font-medium text-green-700 dark:text-green-300">
                {(uploadedDocument.file.size / 1024).toFixed(1)} KB
              </p>
            </div>
            <div className="text-sm bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-4 rounded-xl max-h-32 overflow-y-auto border shadow-inner">
              {uploadedDocument.content.substring(0, 200)}...
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="shadow-lg border-2 hover:shadow-xl transition-all duration-300" style={{ borderColor: 'var(--border)' }}>
      <CardHeader className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20">
        <CardTitle className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-500 text-white">
            <Upload className="h-5 w-5" />
          </div>
          <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent font-bold">
            Upload Document
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
            isDragging 
              ? 'border-blue-400 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/30 dark:to-purple-950/30 scale-105' 
              : 'border-gray-300 hover:border-blue-300 hover:bg-gradient-to-br hover:from-blue-25 hover:to-purple-25'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className={`transition-all duration-300 ${isDragging ? 'scale-110' : ''}`}>
            <Upload className={`h-12 w-12 mx-auto mb-4 ${isDragging ? 'text-blue-500' : 'text-muted-foreground'}`} />
          </div>
          <h3 className="mb-2 font-semibold">Drop your document here</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Supports PDF, TXT files up to 10MB
          </p>
          <input
            type="file"
            accept=".pdf,.txt"
            onChange={handleFileSelect}
            className="hidden"
            id="file-upload"
          />
          <Button asChild className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300">
            <label htmlFor="file-upload" className="cursor-pointer">
              Choose File
            </label>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}