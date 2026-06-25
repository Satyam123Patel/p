"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Play, Copy, Trash2, Loader } from "lucide-react"
import { CodeEditor } from "./code-editor"
import { OutputPanel } from "./output-panel"

const DEFAULT_CODE = `public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}`

export function JavaRunner() {
  const [code, setCode] = useState(DEFAULT_CODE)
  const [output, setOutput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const runCode = async () => {
    setLoading(true)
    setError("")
    setOutput("")

    try {
      const response = await fetch("/api/execute-java", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      })

      const data = await response.json()

      if (!response.ok) {
        setError(data.error || "Failed to execute code")
        return
      }

      setOutput(data.output || "")
      if (data.error) {
        setError(data.error)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred")
    } finally {
      setLoading(false)
    }
  }

  const clearCode = () => {
    setCode(DEFAULT_CODE)
    setOutput("")
    setError("")
  }

  const copyCode = () => {
    navigator.clipboard.writeText(code)
  }

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card p-4">
        <div className="max-w-full mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Java Runner</h1>
            <p className="text-sm text-muted-foreground">Write and execute Java code in your browser</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={copyCode} className="gap-2 bg-transparent">
              <Copy className="h-4 w-4" />
              Copy
            </Button>
            <Button variant="outline" size="sm" onClick={clearCode} className="gap-2 bg-transparent">
              <Trash2 className="h-4 w-4" />
              Clear
            </Button>
            <Button
              onClick={runCode}
              disabled={loading}
              className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
            >
              {loading ? (
                <>
                  <Loader className="h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Run Code
                </>
              )}
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden gap-4 p-4">
        {/* Code Editor */}
        <div className="flex-1 flex flex-col min-w-0">
          <Card className="flex-1 overflow-hidden bg-card border border-border">
            <CodeEditor code={code} onChange={setCode} />
          </Card>
        </div>

        {/* Output Panel */}
        <div className="flex-1 flex flex-col min-w-0">
          <OutputPanel output={output} error={error} loading={loading} />
        </div>
      </div>
    </div>
  )
}
