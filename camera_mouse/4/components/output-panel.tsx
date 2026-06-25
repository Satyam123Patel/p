"use client"

import { Card } from "@/components/ui/card"
import { AlertCircle, CheckCircle } from "lucide-react"

interface OutputPanelProps {
  output: string
  error: string
  loading: boolean
}

export function OutputPanel({ output, error, loading }: OutputPanelProps) {
  const hasContent = output || error

  return (
    <div className="flex flex-col min-h-0">
      <h2 className="text-sm font-semibold text-foreground mb-2">Output</h2>
      <Card className="flex-1 overflow-hidden bg-card border border-border flex flex-col">
        <div className="flex-1 overflow-auto p-4 font-mono text-sm">
          {loading && <div className="text-muted-foreground animate-pulse">Executing code...</div>}

          {error && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-destructive">
                <AlertCircle className="h-4 w-4 flex-shrink-0" />
                <span className="font-semibold">Error</span>
              </div>
              <pre className="text-destructive whitespace-pre-wrap break-words text-xs">{error}</pre>
            </div>
          )}

          {output && !error && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                <CheckCircle className="h-4 w-4 flex-shrink-0" />
                <span className="font-semibold">Success</span>
              </div>
              <pre className="text-foreground whitespace-pre-wrap break-words text-xs">{output}</pre>
            </div>
          )}

          {!hasContent && !loading && (
            <div className="text-muted-foreground text-center py-8">Run your code to see the output here</div>
          )}
        </div>
      </Card>
    </div>
  )
}
