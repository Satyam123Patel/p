"use client";

import { useEffect, useRef } from "react";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

export default function HandTracker() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const handRef = useRef<HandLandmarker | null>(null);
  const runningRef = useRef(false);

  useEffect(() => {
    if (runningRef.current) return;
    runningRef.current = true;

    async function init() {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );

      handRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        },
        runningMode: "VIDEO",
        numHands: 1,
      });

      const video = videoRef.current!;
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;

      await new Promise((resolve) => {
        video.onloadedmetadata = () => resolve(true);
      });

      video.play();
      requestAnimationFrame(() => detect(video));
    }

    async function detect(video: HTMLVideoElement) {
      if (!handRef.current) return;

      if (video.readyState >= 2) {
        const results = await handRef.current.detectForVideo(
          video,
          performance.now()
        );

        console.log(results.landmarks);
      }

      requestAnimationFrame(() => detect(video));
    }

    init();
  }, []);

  return <video ref={videoRef} style={{ display: "none" }} />;
}
