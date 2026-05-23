import { useState } from 'react';
import { Send, Loader2 } from 'lucide-react';
import type { ChatMessage, Session, AnalysisResult } from '../types';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Card } from './ui/card';
import { chatWithAI } from '../utils/ai-service';

interface ChatTabProps {
  session: Session;
  currentAnalysis?: AnalysisResult;
  onDismissMetric?: (metricName: string) => void;
  onRestoreMetric?: (metricName: string) => void;
}

export function ChatTab({
  session,
  currentAnalysis,
  onDismissMetric,
  onRestoreMetric,
}: ChatTabProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const quickActions = [
    'What should I fix first?',
    'Explain my current score',
    'How can I improve?',
  ];

  const handleSend = async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const lowerText = text.toLowerCase();

      if (lowerText.includes('dismiss') || lowerText.includes('ignore')) {
        const metricMatch = currentAnalysis?.metrics.find(m =>
          lowerText.includes(m.name.toLowerCase())
        );
        if (metricMatch && onDismissMetric) {
          onDismissMetric(metricMatch.name);
          const assistantMessage: ChatMessage = {
            role: 'assistant',
            content: `I've dismissed "${metricMatch.name}" for you. This metric will no longer affect your composite score.`,
            timestamp: new Date().toISOString(),
          };
          setMessages(prev => [...prev, assistantMessage]);
          setIsLoading(false);
          return;
        }
      }

      if (lowerText.includes('restore') || lowerText.includes('un-dismiss')) {
        const metricMatch = session.dismissedMetrics.find(name =>
          lowerText.includes(name.toLowerCase())
        );
        if (metricMatch && onRestoreMetric) {
          onRestoreMetric(metricMatch);
          const assistantMessage: ChatMessage = {
            role: 'assistant',
            content: `I've restored "${metricMatch}" for you. This metric will now be included in your composite score.`,
            timestamp: new Date().toISOString(),
          };
          setMessages(prev => [...prev, assistantMessage]);
          setIsLoading(false);
          return;
        }
      }

      const response = await chatWithAI([...messages, userMessage], session, currentAnalysis);

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'I apologize, but I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickAction = (action: string) => {
    handleSend(action);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-16rem)]">
      <div className="flex-1 overflow-y-auto mb-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-8">
            <p className="text-gray-500 mb-4">
              Ask me anything about your diagram quality or how to improve it!
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {quickActions.map((action) => (
                <Button
                  key={action}
                  variant="outline"
                  size="sm"
                  onClick={() => handleQuickAction(action)}
                  disabled={isLoading}
                >
                  {action}
                </Button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, idx) => (
          <Card
            key={idx}
            className={`p-4 ${
              message.role === 'user'
                ? 'bg-blue-50 ml-auto max-w-[80%]'
                : 'bg-gray-50 mr-auto max-w-[80%]'
            }`}
          >
            <div className="text-xs text-gray-500 mb-1">
              {message.role === 'user' ? 'You' : 'Assistant'}
            </div>
            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          </Card>
        ))}

        {isLoading && (
          <Card className="p-4 bg-gray-50 mr-auto max-w-[80%]">
            <div className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm text-gray-600">Thinking...</span>
            </div>
          </Card>
        )}
      </div>

      <div className="border-t pt-4">
        <div className="flex gap-2">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend(input);
              }
            }}
            placeholder="Ask about your diagram quality, request suggestions, or dismiss metrics..."
            className="flex-1 min-h-[80px]"
            disabled={isLoading}
          />
          <Button
            onClick={() => handleSend(input)}
            disabled={!input.trim() || isLoading}
            size="lg"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
