import { useState } from 'react';
import { DocumentUploader } from './components/DocumentUploader';
import { QuestionInput } from './components/QuestionInput';
import { AnswerDisplay } from './components/AnswerDisplay';
import { AnalysisHistory } from './components/AnalysisHistory';
import { DocumentViewer } from './components/DocumentViewer';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { FileText, MessageSquare, History, Eye } from 'lucide-react';
import { Toaster } from './components/ui/sonner';

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

export default function App() {
  const [uploadedDocument, setUploadedDocument] = useState<{ file: File; content: string } | null>(null);
  const [qaHistory, setQAHistory] = useState<QAItem[]>([]);
  const [currentAnswer, setCurrentAnswer] = useState<CurrentAnswer | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [highlightedSections, setHighlightedSections] = useState<string[]>([]);

  const handleDocumentUpload = (file: File, content: string) => {
    setUploadedDocument({ file, content });
    setQAHistory([]);
    setCurrentAnswer(null);
    setHighlightedSections([]);
  };

  const handleRemoveDocument = () => {
    setUploadedDocument(null);
    setQAHistory([]);
    setCurrentAnswer(null);
    setHighlightedSections([]);
  };

  const generateMockAnswer = (question: string): CurrentAnswer => {
    // Mock AI responses based on question type
    const responses = {
      purpose: {
        answer: "Based on the document analysis, the main purpose appears to be providing comprehensive guidelines for technical implementation and operational procedures. The document serves as a reference guide for teams working on system integration and process optimization.",
        confidence: 0.92,
        relevantSections: ["Introduction", "Overview", "Technical Requirements"],
        sources: ["Document header", "Executive summary section", "Conclusion"]
      },
      summary: {
        answer: "The document outlines key technical requirements, implementation guidelines, and best practices for system development. It covers operational procedures, quality standards, and provides step-by-step instructions for successful project execution.",
        confidence: 0.88,
        relevantSections: ["Technical Requirements", "Implementation Guidelines", "Best Practices"],
        sources: ["Main content sections", "Process documentation", "Guidelines chapter"]
      },
      requirements: {
        answer: "The technical requirements include system compatibility standards, performance benchmarks, security protocols, and integration specifications. Minimum hardware requirements and software dependencies are also specified.",
        confidence: 0.95,
        relevantSections: ["Technical Requirements", "System Specifications"],
        sources: ["Requirements section", "Technical appendix"]
      },
      audience: {
        answer: "The target audience includes technical teams, project managers, system administrators, and stakeholders involved in implementation processes. The document is designed for both technical and non-technical readers.",
        confidence: 0.85,
        relevantSections: ["Introduction", "Audience"],
        sources: ["Document introduction", "Scope definition"]
      },
      default: {
        answer: "Based on the document content, I've analyzed the relevant sections and found information that addresses your question. The document provides detailed insights into the topic you've inquired about, with supporting evidence from multiple sections.",
        confidence: 0.78,
        relevantSections: ["Overview", "Implementation"],
        sources: ["Document content analysis"]
      }
    };

    const questionLower = question.toLowerCase();
    let response = responses.default;

    if (questionLower.includes('purpose') || questionLower.includes('main')) {
      response = responses.purpose;
    } else if (questionLower.includes('summary') || questionLower.includes('key points')) {
      response = responses.summary;
    } else if (questionLower.includes('requirements') || questionLower.includes('technical')) {
      response = responses.requirements;
    } else if (questionLower.includes('audience') || questionLower.includes('target')) {
      response = responses.audience;
    }

    return {
      question,
      ...response
    };
  };

  const handleSubmitQuestion = async (question: string) => {
    if (!uploadedDocument) return;

    setIsLoading(true);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1500));

    const answer = generateMockAnswer(question);
    
    const qaItem: QAItem = {
      id: Date.now().toString(),
      question,
      answer: answer.answer,
      confidence: answer.confidence,
      timestamp: new Date(),
      relevantSections: answer.relevantSections
    };

    setCurrentAnswer(answer);
    setQAHistory(prev => [qaItem, ...prev]);
    setIsLoading(false);
  };

  const handleHighlightSections = (sections: string[]) => {
    setHighlightedSections(sections);
  };

  return (
    <div className="min-h-screen" style={{ background: 'var(--background)' }}>
      <div className="container mx-auto p-6 space-y-6">
        {/* Header */}
        <Card className="border-2 shadow-lg" style={{ borderImage: 'var(--primary) 1' }}>
          <CardHeader className="relative overflow-hidden">
            <div className="absolute inset-0 opacity-10" style={{ background: 'var(--primary)' }}></div>
            <CardTitle className="flex items-center gap-3 relative z-10 text-2xl">
              <div className="p-2 rounded-xl shadow-md" style={{ background: 'var(--primary)' }}>
                <FileText className="h-8 w-8 text-white" />
              </div>
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent font-bold">
                Document QA Analysis Platform
              </span>
            </CardTitle>
            <p className="text-muted-foreground relative z-10 text-lg">
              Upload documents and ask questions to get AI-powered analysis and insights.
            </p>
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
                <QuestionInput
                  onSubmitQuestion={handleSubmitQuestion}
                  isLoading={isLoading}
                  disabled={!uploadedDocument}
                />
                <AnswerDisplay
                  currentAnswer={currentAnswer}
                  onHighlightSections={handleHighlightSections}
                />
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