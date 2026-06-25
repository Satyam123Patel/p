"use client"

import type React from "react"

import { useEffect, useRef } from "react"

interface CodeEditorProps {
  code: string
  onChange: (code: string) => void
}

export function CodeEditor({ code, onChange }: CodeEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return

    const syncScroll = () => {
      const lineNumbers = document.querySelector(".line-numbers")
      if (lineNumbers) {
        lineNumbers.scrollTop = textarea.scrollTop
      }
    }

    textarea.addEventListener("scroll", syncScroll)
    return () => textarea.removeEventListener("scroll", syncScroll)
  }, [])

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value)
  }

  const lineCount = code.split("\n").length
  const lineNumbers = Array.from({ length: lineCount }, (_, i) => i + 1)

  return (
    <div className="flex h-full bg-foreground/5">
      <div className="line-numbers w-12 bg-muted/50 text-muted-foreground text-right pr-3 pt-3 overflow-hidden select-none font-mono text-sm leading-6 border-r border-border">
        {lineNumbers.map((num) => (
          <div key={num}>{num}</div>
        ))}
      </div>
      <textarea
        ref={textareaRef}
        value={code}
        onChange={handleChange}
        className="flex-1 p-3 bg-card text-foreground font-mono text-sm leading-6 resize-none focus:outline-none border-none"
        spellCheck="false"
      />
    </div>
  )
}
