import { FC } from 'react';

declare const QuestionInput: FC<{
  onAsk: (question: string) => void;
  disabled: boolean;
}>;

export default QuestionInput;
