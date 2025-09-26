import React, { useState, useEffect } from 'react';
import { DocumentUploader } from './components/DocumentUploader';
import { QuestionInput } from './components/QuestionInput';
import { AnswerDisplay } from './components/AnswerDisplay';
import { AnalysisHistory } from './components/AnalysisHistory';
import { DocumentViewer } from './components/DocumentViewer';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { FileText, MessageSquare, History, Eye, Loader2 } from 'lucide-react';
import { Toaster } from './components/ui/sonner';
import { uploadFiles, askQuestion, checkHealth, resetDatabase, AskResponse } from './services/api';
import { ResetButton } from './components/ResetButton';

interface QAItem {
  id: string;
  question: string;
  answer: string;
  confidence: number;
  timestamp: Date;
  relevantSections: string[];
}

interface CurrentAnswer {
  question: string;
  answer: string;
  confidence: number;
  relevantSections: string[];
  sources: string[];
}

// Import the component types
import type { FC } from 'react';

// Define the QuestionInput props type
type QuestionInputProps = {
  onSubmitQuestion: (question: string) => void;
  isLoading: boolean;
  disabled: boolean;
};

// Define the AnswerDisplay props type
type AnswerDisplayProps = {
  currentAnswer: {
    question: string;
    answer: string;
    confidence: number;
    relevantSections: string[];
    sources: string[];
  } | null;
  onHighlightSections: (sections: string[]) => void;
};

export default function App() {
  const [uploadedDocument, setUploadedDocument] = useState<{ file: File; content: string } | null>(null);
  const [qaHistory, setQAHistory] = useState<QAItem[]>([]);
  const [currentAnswer, setCurrentAnswer] = useState<CurrentAnswer | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [highlightedSections, setHighlightedSections] = useState<string[]>([]);

  const handleDocumentUpload = async (file: File, content: string) => {
    try {
      setIsLoading(true);
      const response = await uploadFiles([file]);
      if (response.error) {
        console.error('Upload failed:', response.error);
        // Show error to user (you can replace this with your preferred error handling)
        alert(`Upload failed: ${response.error}`);
        return;
      }
      setUploadedDocument({ file, content });
      setQAHistory([]);
      setCurrentAnswer(null);
      setHighlightedSections([]);
      console.log('Document uploaded successfully');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      console.error('Error uploading document:', errorMessage);
      alert(`Error uploading document: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRemoveDocument = () => {
    setUploadedDocument(null);
    setQAHistory([]);
    setCurrentAnswer(null);
    setHighlightedSections([]);
  };

  const askQuestionToBackend = async (question: string): Promise<AskResponse> => {
    const response = await askQuestion(question);
    if (response.error) {
      throw new Error(response.error);
    }
    return response.data!;
  };

  const handleAskQuestion = async (question: string) => {
    if (!question.trim() || !uploadedDocument) return;

    setIsLoading(true);

    try {
      const answer = await askQuestionToBackend(question);

      // Update current answer
      const currentAnswer = {
        question: answer.question,
        answer: answer.answer,
        confidence: 0.95, // You might want to get this from the backend
        relevantSections: answer.sources,
        sources: answer.sources
      };

      setCurrentAnswer(currentAnswer);

      // Add to history
      const newQAItem: QAItem = {
        id: Date.now().toString(),
        question: currentAnswer.question,
        answer: currentAnswer.answer,
        confidence: currentAnswer.confidence,
        timestamp: new Date(),
        relevantSections: currentAnswer.relevantSections
      };

      setQAHistory(prev => [newQAItem, ...prev]);
      setHighlightedSections(currentAnswer.sources);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      console.error('Error getting answer:', errorMessage);
      alert(`Error getting answer: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleHighlightSections = (sections: string[]) => {
    setHighlightedSections(sections);
  };

  const handleResetComplete = () => {
    // Clear all local state after successful reset
    setUploadedDocument(null);
    setQAHistory([]);
    setCurrentAnswer(null);
    setHighlightedSections([]);
  };

  return (
    <div className="min-h-screen" style={{ background: 'var(--background)' }}>
      <div className="container mx-auto p-6 space-y-6">
        {/* Header */}
        <Card className="border-2 shadow-lg" style={{ borderImage: 'var(--primary) 1' }}>
          <CardHeader className="relative overflow-hidden">
            <div className="absolute inset-0 opacity-10" style={{ background: 'var(--primary)' }}></div>
            <div className="flex items-start justify-between relative z-10">
              <div className="flex-1">
                <CardTitle className="flex items-center gap-3 text-2xl">
                  <div className="p-2 rounded-xl shadow-md" style={{ background: 'var(--primary)' }}>
                    <FileText className="h-8 w-8 text-white" />
                  </div>
                  <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent font-bold">
                    Document QA Analysis Platform
                  </span>
                </CardTitle>
                <p className="text-muted-foreground text-lg mt-2">
                  Upload documents and ask questions to get AI-powered analysis and insights.
                </p>
              </div>
              {uploadedDocument && (
                <div className="ml-4">
                  <ResetButton
                    onResetComplete={handleResetComplete}
                    disabled={isLoading}
                  />
                </div>
              )}
            </div>
          </CardHeader>
        </Card>

        {/* Document Upload */}
        <DocumentUploader
          onDocumentUpload={handleDocumentUpload}
          uploadedDocument={uploadedDocument}
          onRemoveDocument={handleRemoveDocument}
        />

        {uploadedDocument && (
          <Tabs defaultValue="analysis" className="space-y-6">
            <TabsList className="grid w-full grid-cols-3 bg-white/80 backdrop-blur-sm shadow-lg border-2" style={{ borderColor: 'var(--border)' }}>
              <TabsTrigger value="analysis" className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-500 data-[state=active]:to-purple-500 data-[state=active]:text-white transition-all duration-300">
                <MessageSquare className="h-4 w-4" />
                Analysis
              </TabsTrigger>
              <TabsTrigger value="document" className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500 data-[state=active]:to-teal-500 data-[state=active]:text-white transition-all duration-300">
                <Eye className="h-4 w-4" />
                Document
              </TabsTrigger>
              <TabsTrigger value="history" className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-orange-500 data-[state=active]:to-red-500 data-[state=active]:text-white transition-all duration-300">
                <History className="h-4 w-4" />
                History
              </TabsTrigger>
            </TabsList>

            <TabsContent value="analysis" className="space-y-6">
              <div className="grid lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <MessageSquare className="h-5 w-5" />
                      Ask a Question
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <QuestionInput 
                      onSubmitQuestion={handleAskQuestion}
                      isLoading={isLoading}
                      disabled={!uploadedDocument || isLoading} 
                    />
                    <div className="mt-6">
                      <AnswerDisplay 
                        currentAnswer={currentAnswer}
                        onHighlightSections={setHighlightedSections}
                      />
                    </div>
                    {isLoading && (
                      <div className="mt-4 flex flex-col items-center justify-center p-8 text-center">
                        <Loader2 className="h-8 w-8 animate-spin text-primary mb-2" />
                        <p className="text-muted-foreground">Analyzing document and generating answer...</p>
                        <p className="text-sm text-muted-foreground mt-2">This may take a moment</p>
                      </div>
                    )}
                    {!currentAnswer && !isLoading && (
                      <div className="mt-8 text-center text-muted-foreground">
                        {uploadedDocument 
                          ? "Ask a question about the uploaded document"
                          : "Upload a document to start asking questions"}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="document">
              <DocumentViewer
                document={uploadedDocument}
                highlightedSections={highlightedSections}
              />
            </TabsContent>

            <TabsContent value="history">
              <AnalysisHistory qaHistory={qaHistory} />
            </TabsContent>
          </Tabs>
        )}
      </div>
      <Toaster />
    </div>
  );
}