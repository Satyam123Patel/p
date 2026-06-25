import HandTracker from "@/components/hand-tracker";

export const metadata = {
  title: "Camera Mouse — Gesture Control",
  description:
    "Control your cursor with hand gestures using your webcam. Powered by MediaPipe AI.",
};

export default function Home() {
  return (
    <main>
      <HandTracker />
    </main>
  );
}
