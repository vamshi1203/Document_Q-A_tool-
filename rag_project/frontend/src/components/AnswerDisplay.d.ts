import { FC } from 'react';

declare const AnswerDisplay: FC<{
  question: string;
  answer: string;
  confidence: number;
  sources: string[];
}>;

export default AnswerDisplay;
