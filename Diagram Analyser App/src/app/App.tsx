import { useState, useEffect, useRef } from 'react';
import { Upload, Plus, FileText, Download, Loader2, X } from 'lucide-react';
import type { Session, DiagramType, AnalysisResult, Severity, MetricThreshold, MetricName } from './types';
import { DEFAULT_THRESHOLDS } from './types';
import {
  getAllSessions,
  createSession,
  getSession,
  updateSession,
  deleteSession,
  downloadSessionJson,
} from './utils/session-manager';
import { analyzeImage, recalculateScores } from './utils/analysis-engine';
import { apiFetch } from './utils/api-client';
import { AnalysisTab } from './components/AnalysisTab';
import { VersionHistoryTab } from './components/VersionHistoryTab';
import { ChatTab } from './components/ChatTab';
import { Button } from './components/ui/button';
import { Card } from './components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './components/ui/dialog';
import { ScrollArea } from './components/ui/scroll-area';
import { Badge } from './components/ui/badge';

// React display name → Python metric key (for threshold PATCH calls)
const REACT_NAME_TO_PYTHON_KEY: Record<string, string> = {
  'Label Readability':       'label_readability',
  'Label Area':              'label_area',
  'Overlap (Crowding)':      'overlap_metrics',
  'Edge Clearance':          'edge_clearance',
  'Font Hierarchy':          'font_hierarchy',
  'Container Utilisation':   'container_utilization',
  'Isolated Boxes':          'isolated_boxes',
  'Brevity':                 'brevity',
  'Whitespace Distribution': 'whitespace_distribution',
  'Color Harmony':           'color_harmony',
  'Label Contrast':          'label_contrast',
  'Cognitive Chunk Density': 'cognitive_chunk_density',
  'Orientation Consistency': 'orientation_consistency',
};

export default function App() {
  const [sessions, setSessions]                 = useState<Session[]>([]);
  const [currentSession, setCurrentSession]     = useState<Session | null>(null);
  const [isNewSessionOpen, setIsNewSessionOpen] = useState(false);
  const [newSessionName, setNewSessionName]     = useState('');
  const [newSessionType, setNewSessionType]     = useState<DiagramType>('system-design');
  const [activeTab, setActiveTab]               = useState('analysis');
  const [isAnalyzing, setIsAnalyzing]           = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getAllSessions().then((allSessions) => {
      setSessions(allSessions);
      if (allSessions.length > 0) {
        setCurrentSession(allSessions[allSessions.length - 1]);
      }
    }).catch(console.error);
  }, []);

  const refreshSessions = async (selectId?: string) => {
    const allSessions = await getAllSessions();
    setSessions(allSessions);
    if (selectId) {
      const selected = allSessions.find(s => s.id === selectId) ?? null;
      setCurrentSession(selected);
    }
  };

  const handleDeleteSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    await deleteSession(sessionId);
    const allSessions = await getAllSessions();
    setSessions(allSessions);
    if (currentSession?.id === sessionId) {
      setCurrentSession(allSessions.length > 0 ? allSessions[allSessions.length - 1] : null);
    }
  };

  const handleCreateSession = async () => {
    if (!newSessionName.trim()) return;

    const session = await createSession(newSessionName, newSessionType);
    await refreshSessions(session.id);
    setNewSessionName('');
    setIsNewSessionOpen(false);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !currentSession) return;
    // Reset input so the same file can be re-uploaded
    event.target.value = '';
    await processImage(file);
  };

  const processImage = async (file: File) => {
    if (!currentSession) return;

    setIsAnalyzing(true);
    try {
      // Server runs analysis and appends the new version to the session
      await analyzeImage(file, currentSession.id);
      // Re-fetch the full session so the UI reflects the new version
      const updated = await getSession(currentSession.id);
      if (updated) {
        setCurrentSession(updated);
        await refreshSessions();
      }
      setActiveTab('analysis');
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDismissMetric = async (metricName: string) => {
    if (!currentSession) return;
    if (currentSession.dismissedMetrics.includes(metricName)) return;

    const updatedSession: Session = {
      ...currentSession,
      dismissedMetrics: [...currentSession.dismissedMetrics, metricName],
    };

    if (currentSession.versions.length > 0) {
      updatedSession.versions = currentSession.versions.map((version, idx) => {
        if (idx === currentSession.versions.length - 1) {
          const { metrics, compositeScore } = recalculateScores(
            version.metrics,
            updatedSession.dismissedMetrics,
            updatedSession.customThresholds,
          );
          return {
            ...version,
            metrics,
            compositeScore,
            criticalCount: metrics.filter(m => m.severity === 'critical' && !m.isDismissed).length,
            warningCount:  metrics.filter(m => m.severity === 'warning'  && !m.isDismissed).length,
          };
        }
        return version;
      });
    }

    setCurrentSession(updatedSession);
    await updateSession(updatedSession);
    await refreshSessions();
  };

  const handleRestoreMetric = async (metricName: string) => {
    if (!currentSession) return;

    const updatedSession: Session = {
      ...currentSession,
      dismissedMetrics: currentSession.dismissedMetrics.filter(name => name !== metricName),
    };

    if (currentSession.versions.length > 0) {
      updatedSession.versions = currentSession.versions.map((version, idx) => {
        if (idx === currentSession.versions.length - 1) {
          const { metrics, compositeScore } = recalculateScores(
            version.metrics,
            updatedSession.dismissedMetrics,
            updatedSession.customThresholds,
          );
          return {
            ...version,
            metrics,
            compositeScore,
            criticalCount: metrics.filter(m => m.severity === 'critical' && !m.isDismissed).length,
            warningCount:  metrics.filter(m => m.severity === 'warning'  && !m.isDismissed).length,
          };
        }
        return version;
      });
    }

    setCurrentSession(updatedSession);
    await updateSession(updatedSession);
    await refreshSessions();
  };

  const handleUpdateSeverity = async (metricName: string, newSeverity: Severity) => {
    if (!currentSession || currentSession.versions.length === 0) return;

    const currentMetric = currentSession.versions[currentSession.versions.length - 1].metrics.find(
      m => m.name === metricName,
    );
    if (!currentMetric) return;

    const score = currentMetric.score;
    const defaultThreshold = DEFAULT_THRESHOLDS[metricName as MetricName];
    let newThreshold: MetricThreshold;

    switch (newSeverity) {
      case 'critical':
        newThreshold = {
          critical: Math.max(score + 1, defaultThreshold.critical),
          warning:  defaultThreshold.warning,
        };
        break;
      case 'warning':
        newThreshold = {
          critical: Math.min(score - 1, defaultThreshold.critical),
          warning:  Math.max(score + 1, defaultThreshold.warning),
        };
        break;
      case 'pass':
        newThreshold = {
          critical: defaultThreshold.critical,
          warning:  Math.min(score - 1, defaultThreshold.warning),
        };
        break;
    }

    const updatedSession: Session = {
      ...currentSession,
      customThresholds: {
        ...currentSession.customThresholds,
        [metricName]: newThreshold,
      },
    };

    updatedSession.versions = currentSession.versions.map((version, idx) => {
      if (idx === currentSession.versions.length - 1) {
        const { metrics, compositeScore } = recalculateScores(
          version.metrics,
          updatedSession.dismissedMetrics,
          updatedSession.customThresholds,
        );
        return {
          ...version,
          metrics,
          compositeScore,
          criticalCount: metrics.filter(m => m.severity === 'critical' && !m.isDismissed).length,
          warningCount:  metrics.filter(m => m.severity === 'warning'  && !m.isDismissed).length,
        };
      }
      return version;
    });

    setCurrentSession(updatedSession);

    // Persist dismiss list / thresholds to the server
    await updateSession(updatedSession);
    await refreshSessions();

    // Fire-and-forget: teach the Python threshold learner about the reclassification
    const pyKey = REACT_NAME_TO_PYTHON_KEY[metricName];
    if (pyKey) {
      apiFetch(`/api/sessions/${currentSession.id}/metrics/${pyKey}/severity`, {
        method: 'PATCH',
        body: JSON.stringify({
          oldSeverity:  currentMetric.severity,
          newSeverity,
          currentScore: score,
        }),
      }).catch(() => { /* non-critical */ });
    }
  };

  const currentAnalysis  = currentSession?.versions[currentSession.versions.length - 1];
  const previousAnalysis = currentSession?.versions[currentSession.versions.length - 2];

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r flex flex-col">
        <div className="p-4 border-b">
          <h1 className="text-xl font-bold mb-4">Diagram Analyser</h1>
          <Button className="w-full" onClick={() => setIsNewSessionOpen(true)}>
            <Plus className="w-4 h-4 mr-2" />
            New Session
          </Button>
        </div>

        <ScrollArea className="flex-1">
          <div className="p-4 space-y-2">
            {sessions.length === 0 && (
              <p className="text-sm text-gray-500 text-center py-8">
                No sessions yet. Create one to get started!
              </p>
            )}
            {sessions.map((session) => (
              <Card
                key={session.id}
                className={`p-3 cursor-pointer transition-colors relative group ${
                  currentSession?.id === session.id ? 'bg-blue-50 border-blue-300' : 'hover:bg-gray-50'
                }`}
                onClick={() => {
                  getSession(session.id).then(s => {
                    if (s) setCurrentSession(s);
                  });
                  setActiveTab('analysis');
                }}
              >
                <button
                  className="absolute top-2 right-2 p-0.5 rounded transition-opacity text-gray-400 cursor-pointer"
                  onClick={(e) => handleDeleteSession(e, session.id)}
                  title="Delete session"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
                <div className="font-medium text-sm mb-1 pr-5">{session.name}</div>
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Badge variant="outline" className="text-xs">
                    {session.diagramType === 'system-design' ? 'System Design' : 'Timeline/Roadmap'}
                  </Badge>
                  <span>{session.versions.length} version{session.versions.length !== 1 ? 's' : ''}</span>
                </div>
                {session.versions.length > 0 && (
                  <div className="text-xs text-gray-600 mt-1">
                    Score: {session.versions[session.versions.length - 1].compositeScore}/100
                  </div>
                )}
              </Card>
            ))}
          </div>
        </ScrollArea>

        <div className="p-4 border-t space-y-2">
          {currentSession && (
            <Button
              variant="outline"
              className="w-full"
              onClick={() => downloadSessionJson(currentSession)}
            >
              <Download className="w-4 h-4 mr-2" />
              Export Session
            </Button>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col">
        {!currentSession ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <FileText className="w-16 h-16 mx-auto text-gray-400 mb-4" />
              <h2 className="text-xl font-semibold mb-2">No Session Selected</h2>
              <p className="text-gray-500">
                Create a new session from the sidebar to begin analyzing diagrams.
              </p>
            </div>
          </div>
        ) : (
          <>
            <div className="bg-white border-b p-4">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-xl font-semibold">{currentSession.name}</h2>
                  <p className="text-sm text-gray-500">
                    {currentSession.diagramType === 'system-design' ? 'System Design Diagram' : 'Timeline/Roadmap'}
                  </p>
                </div>
                <div className="flex gap-2">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleFileUpload}
                  />
                  <Button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Upload className="w-4 h-4 mr-2" />
                        Upload New Version
                      </>
                    )}
                  </Button>
                </div>
              </div>

              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList>
                  <TabsTrigger value="analysis">Analysis</TabsTrigger>
                  <TabsTrigger value="history" disabled={currentSession.versions.length < 2}>
                    Version History
                  </TabsTrigger>
                  <TabsTrigger value="chat">Chat</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            <ScrollArea className="flex-1 p-6">
              {currentSession.versions.length === 0 ? (
                <div className="flex items-center justify-center h-full">
                  {isAnalyzing ? (
                    <div className="text-center">
                      <Loader2 className="w-16 h-16 mx-auto text-blue-500 mb-4 animate-spin" />
                      <h3 className="text-lg font-semibold mb-2">Analysing your diagram…</h3>
                      <p className="text-gray-500">This may take a few seconds. Please wait.</p>
                    </div>
                  ) : (
                    <div className="text-center">
                      <Upload className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                      <h3 className="text-lg font-semibold mb-2">No Diagrams Yet</h3>
                      <p className="text-gray-500 mb-4">
                        Upload your first diagram to get started with analysis.
                      </p>
                      <Button onClick={() => fileInputRef.current?.click()}>
                        <Upload className="w-4 h-4 mr-2" />
                        Upload Diagram
                      </Button>
                    </div>
                  )}
                </div>
              ) : (
                <Tabs value={activeTab}>
                  <TabsContent value="analysis">
                    {currentAnalysis && (
                      <AnalysisTab
                        analysis={currentAnalysis}
                        previousAnalysis={previousAnalysis}
                        onDismissMetric={handleDismissMetric}
                        onRestoreMetric={handleRestoreMetric}
                        onUpdateSeverity={handleUpdateSeverity}
                      />
                    )}
                  </TabsContent>
                  <TabsContent value="history">
                    <VersionHistoryTab versions={currentSession.versions} />
                  </TabsContent>
                  <TabsContent value="chat">
                    <ChatTab
                      session={currentSession}
                      currentAnalysis={currentAnalysis}
                      onDismissMetric={handleDismissMetric}
                      onRestoreMetric={handleRestoreMetric}
                    />
                  </TabsContent>
                </Tabs>
              )}
            </ScrollArea>
          </>
        )}
      </div>

      {/* New Session dialog */}
      <Dialog open={isNewSessionOpen} onOpenChange={setIsNewSessionOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Session</DialogTitle>
            <DialogDescription>
              Start a new diagram analysis session. Choose the diagram type that matches your content.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <Label className='py-2' htmlFor="session-name">Session Name</Label>
              <Input
                id="session-name"
                placeholder="e.g., My System Architecture"
                value={newSessionName}
                onChange={(e) => setNewSessionName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCreateSession()}
              />
            </div>
            <div>
              <Label className='py-2' htmlFor="diagram-type">Diagram Type</Label>
              <Select
                value={newSessionType}
                onValueChange={(value) => setNewSessionType(value as DiagramType)}
              >
                <SelectTrigger id="diagram-type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="system-design">System Design</SelectItem>
                  <SelectItem value="timeline-roadmap">Timeline/Roadmap</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsNewSessionOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateSession} disabled={!newSessionName.trim()}>
              Create Session
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
