import HandTracker from "@/components/hand-tracker";

export const metadata = {
  title: "Hand Tracker - Gesture Control",
  description: "Control your cursor with hand gestures using your webcam",
}

export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      <HandTracker />
    </main>
  )
}
