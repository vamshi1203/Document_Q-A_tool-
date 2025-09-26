import React, { useState } from 'react';
import { Button } from './ui/button';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from './ui/alert-dialog';
import { RotateCcw, Loader2 } from 'lucide-react';
import { resetDatabase } from '../services/api';
import { toast } from 'sonner';

interface ResetButtonProps {
  onResetComplete: () => void;
  disabled?: boolean;
}

export const ResetButton: React.FC<ResetButtonProps> = ({ onResetComplete, disabled = false }) => {
  const [isResetting, setIsResetting] = useState(false);

  const handleReset = async () => {
    setIsResetting(true);

    try {
      const response = await resetDatabase();

      if (response.error) {
        toast.error(`Reset failed: ${response.error}`);
        return;
      }

      toast.success('Database reset successfully! All documents and data have been cleared.');
      onResetComplete();

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      toast.error(`Error resetting database: ${errorMessage}`);
    } finally {
      setIsResetting(false);
    }
  };

  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          disabled={disabled || isResetting}
          className="text-red-600 border-red-200 hover:bg-red-50 hover:border-red-300 transition-colors"
        >
          {isResetting ? (
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <RotateCcw className="h-4 w-4 mr-2" />
          )}
          Reset Database
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="flex items-center gap-2 text-red-600">
            <RotateCcw className="h-5 w-5" />
            Reset Database
          </AlertDialogTitle>
          <AlertDialogDescription>
            This action will permanently delete all uploaded documents and clear the vector database.
            All questions and answers will be lost. This cannot be undone.
            <br />
            <br />
            Are you sure you want to continue?
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isResetting}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={handleReset}
            disabled={isResetting}
            className="bg-red-600 hover:bg-red-700 text-white"
          >
            {isResetting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Resetting...
              </>
            ) : (
              'Yes, Reset Database'
            )}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};