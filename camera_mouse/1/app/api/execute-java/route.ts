import { type NextRequest, NextResponse } from "next/server"

// Using JDoodle API for Java execution
const JDOODLE_CLIENT_ID = process.env.JDOODLE_CLIENT_ID || "demo"
const JDOODLE_CLIENT_SECRET = process.env.JDOODLE_CLIENT_SECRET || "demo"

export async function POST(request: NextRequest) {
  try {
    const { code } = await request.json()

    if (!code || typeof code !== "string") {
      return NextResponse.json({ error: "No code provided" }, { status: 400 })
    }

    const response = await fetch("https://api.jdoodle.com/v1/execute", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        clientId: JDOODLE_CLIENT_ID,
        clientSecret: JDOODLE_CLIENT_SECRET,
        script: code,
        language: "java",
        versionIndex: "4",
      }),
    })

    const data = await response.json()

    if (data.error) {
      return NextResponse.json({ error: data.error, output: data.output || "" }, { status: 400 })
    }

    return NextResponse.json({
      output: data.output || "",
      error: data.compileStatus === "Success" ? "" : data.output,
    })
  } catch (error) {
    console.error("[v0] Error executing Java code:", error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Failed to execute code",
      },
      { status: 500 },
    )
  }
}
