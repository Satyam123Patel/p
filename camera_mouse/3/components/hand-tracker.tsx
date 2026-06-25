"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

interface Point {
  x: number;
  y: number;
}

type GestureMode = "idle" | "move" | "click" | "drag" | "paused";

const LANDMARK_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // index
  [0, 9], [9, 10], [10, 11], [11, 12],  // middle
  [0, 13], [13, 14], [14, 15], [15, 16],// ring
  [0, 17], [17, 18], [18, 19], [19, 20],// pinky
  [5, 9], [9, 13], [13, 17],            // palm
];

// Smoothing factor (0-1, lower = smoother)
const SMOOTHING = 0.3;
const CLICK_COOLDOWN = 500; // ms
const DRAG_COOLDOWN = 100; // ms
const MODE_SWITCH_DELAY = 150; // ms - prevent flickering between modes

export default function HandTracker() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const handRef = useRef<HandLandmarker | null>(null);
  const runningRef = useRef(false);
  const smoothPosRef = useRef<Point>({ x: 0.5, y: 0.5 });
  const lastClickRef = useRef(0);
  const lastDragRef = useRef(0);
  const animationRef = useRef<number>(0);
  const isDraggingRef = useRef(false);
  const dragStartPosRef = useRef<Point>({ x: 0, y: 0 });
  const lastModeRef = useRef<GestureMode>("idle");
  const modeStableTimeRef = useRef(0);
  const stableModeRef = useRef<GestureMode>("idle");

  const [isLoading, setIsLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState("Initializing...");
  const [isTracking, setIsTracking] = useState(false);
  const [cursorPos, setCursorPos] = useState<Point>({ x: 50, y: 50 });
  const [mode, setMode] = useState<GestureMode>("idle");
  const [fingerCount, setFingerCount] = useState(0);
  const [sensitivity, setSensitivity] = useState(1.8);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [showCamera, setShowCamera] = useState(true);
  const [clickCount, setClickCount] = useState(0);
  const [dragCount, setDragCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [handDetected, setHandDetected] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [clickFlash, setClickFlash] = useState(false);

  const fpsCounterRef = useRef({ frames: 0, lastTime: performance.now() });
  const sensitivityRef = useRef(sensitivity);
  sensitivityRef.current = sensitivity;

  // Count extended fingers
  const countFingers = useCallback(
    (landmarks: { x: number; y: number; z: number }[]): number => {
      let count = 0;

      // Thumb: compare tip x to ip x (different logic since thumb moves sideways)
      // For right hand (mirrored in camera): thumb tip (4) vs thumb IP (3)
      const thumbTip = landmarks[4];
      const thumbIP = landmarks[3];
      const wrist = landmarks[0];
      // Determine if hand is facing left or right based on wrist-to-pinky direction
      const isLeftHand = landmarks[17].x > wrist.x;
      if (isLeftHand) {
        if (thumbTip.x > thumbIP.x) count++;
      } else {
        if (thumbTip.x < thumbIP.x) count++;
      }

      // Index: tip (8) vs PIP (6)
      if (landmarks[8].y < landmarks[6].y) count++;

      // Middle: tip (12) vs PIP (10)
      if (landmarks[12].y < landmarks[10].y) count++;

      // Ring: tip (16) vs PIP (14)
      if (landmarks[16].y < landmarks[14].y) count++;

      // Pinky: tip (20) vs PIP (18)
      if (landmarks[20].y < landmarks[18].y) count++;

      return count;
    },
    []
  );

  // Determine gesture mode from finger count with stabilization
  const getStableMode = useCallback(
    (fingers: number, now: number): GestureMode => {
      let newMode: GestureMode;

      if (fingers === 1) {
        newMode = "move";
      } else if (fingers === 2) {
        newMode = "click";
      } else if (fingers === 3) {
        newMode = "drag";
      } else if (fingers >= 5) {
        newMode = "paused";
      } else {
        newMode = "idle";
      }

      // Stabilize mode switching to prevent flickering
      if (newMode !== lastModeRef.current) {
        lastModeRef.current = newMode;
        modeStableTimeRef.current = now;
        return stableModeRef.current; // Return previous stable mode while waiting
      }

      // Only switch after mode has been stable for MODE_SWITCH_DELAY
      if (now - modeStableTimeRef.current >= MODE_SWITCH_DELAY) {
        stableModeRef.current = newMode;
      }

      return stableModeRef.current;
    },
    []
  );

  const drawLandmarks = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      landmarks: { x: number; y: number }[],
      width: number,
      height: number,
      currentMode: GestureMode
    ) => {
      ctx.clearRect(0, 0, width, height);

      // Color scheme based on mode
      let modeColor = "rgba(0, 200, 255, 0.5)";
      if (currentMode === "move") modeColor = "rgba(0, 229, 255, 0.5)";
      else if (currentMode === "click") modeColor = "rgba(255, 68, 102, 0.5)";
      else if (currentMode === "drag") modeColor = "rgba(255, 170, 0, 0.5)";
      else if (currentMode === "paused") modeColor = "rgba(100, 100, 120, 0.3)";

      // Draw connections
      for (const [start, end] of LANDMARK_CONNECTIONS) {
        const startPoint = landmarks[start];
        const endPoint = landmarks[end];

        ctx.beginPath();
        ctx.moveTo(startPoint.x * width, startPoint.y * height);
        ctx.lineTo(endPoint.x * width, endPoint.y * height);
        ctx.strokeStyle = modeColor;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw landmark points
      for (let i = 0; i < landmarks.length; i++) {
        const point = landmarks[i];
        const x = point.x * width;
        const y = point.y * height;

        ctx.beginPath();
        const isFingertip = [4, 8, 12, 16, 20].includes(i);
        ctx.arc(x, y, isFingertip ? 5 : 3, 0, Math.PI * 2);

        if (isFingertip) {
          if (currentMode === "click") {
            ctx.fillStyle = "rgba(255, 68, 102, 0.9)";
            ctx.shadowColor = "rgba(255, 68, 102, 0.6)";
          } else if (currentMode === "drag") {
            ctx.fillStyle = "rgba(255, 170, 0, 0.9)";
            ctx.shadowColor = "rgba(255, 170, 0, 0.6)";
          } else if (currentMode === "move") {
            ctx.fillStyle = "rgba(0, 229, 255, 0.9)";
            ctx.shadowColor = "rgba(0, 229, 255, 0.6)";
          } else {
            ctx.fillStyle = "rgba(100, 100, 120, 0.5)";
            ctx.shadowColor = "transparent";
          }
          ctx.shadowBlur = 10;
        } else {
          ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
          ctx.shadowBlur = 0;
        }
        ctx.fill();
      }
      ctx.shadowBlur = 0;
    },
    []
  );

  useEffect(() => {
    if (runningRef.current) return;
    runningRef.current = true;

    async function init() {
      try {
        setLoadingMessage("Loading AI vision model...");

        // Suppress harmless TF Lite info messages that trigger Next.js error overlay
        const origConsoleError = console.error;
        console.error = (...args: unknown[]) => {
          const msg = String(args[0]);
          if (msg.includes("TensorFlow Lite") || msg.includes("XNNPACK")) return;
          origConsoleError.apply(console, args);
        };

        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        setLoadingMessage("Creating hand detector...");
        handRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
        });

        // Restore console.error after model loaded
        console.error = origConsoleError;

        setLoadingMessage("Starting camera...");
        const video = videoRef.current!;
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
        });
        video.srcObject = stream;

        await new Promise((resolve) => {
          video.onloadedmetadata = () => resolve(true);
        });

        await video.play();

        // Set canvas dimensions
        if (canvasRef.current) {
          canvasRef.current.width = video.videoWidth;
          canvasRef.current.height = video.videoHeight;
        }
        if (overlayCanvasRef.current) {
          overlayCanvasRef.current.width = video.videoWidth;
          overlayCanvasRef.current.height = video.videoHeight;
        }

        setIsLoading(false);
        setIsTracking(true);

        const detect = () => {
          if (!handRef.current || !videoRef.current) return;
          const v = videoRef.current;

          // FPS counter
          fpsCounterRef.current.frames++;
          const now = performance.now();
          if (now - fpsCounterRef.current.lastTime >= 1000) {
            setFps(fpsCounterRef.current.frames);
            fpsCounterRef.current.frames = 0;
            fpsCounterRef.current.lastTime = now;
          }

          if (v.readyState >= 2) {
            // Draw camera feed to canvas (mirrored)
            const camCtx = canvasRef.current?.getContext("2d");
            if (camCtx && canvasRef.current) {
              camCtx.save();
              camCtx.scale(-1, 1);
              camCtx.drawImage(v, -canvasRef.current.width, 0);
              camCtx.restore();
            }

            const results = handRef.current.detectForVideo(v, now);

            if (results.landmarks && results.landmarks.length > 0) {
              setHandDetected(true);
              const landmarks = results.landmarks[0];

              // Mirror landmarks for display
              const mirroredLandmarks = landmarks.map((l) => ({
                ...l,
                x: 1 - l.x,
              }));

              // Count fingers and determine mode
              const fingers = countFingers(landmarks);
              setFingerCount(fingers);
              const currentMode = getStableMode(fingers, now);
              setMode(currentMode);

              // Get index finger tip for cursor position
              const indexTip = mirroredLandmarks[8];

              // ==========================================
              // MODE: MOVE (1 finger)
              // ==========================================
              if (currentMode === "move") {
                // Release drag if was dragging
                if (isDraggingRef.current) {
                  const el = document.elementFromPoint(
                    smoothPosRef.current.x * window.innerWidth,
                    smoothPosRef.current.y * window.innerHeight
                  );
                  if (el) {
                    el.dispatchEvent(new MouseEvent("mouseup", {
                      bubbles: true,
                      clientX: smoothPosRef.current.x * window.innerWidth,
                      clientY: smoothPosRef.current.y * window.innerHeight,
                    }));
                  }
                  isDraggingRef.current = false;
                }

                // Move cursor
                const sens = sensitivityRef.current;
                const rawX = 0.5 + (indexTip.x - 0.5) * sens;
                const rawY = 0.5 + (indexTip.y - 0.5) * sens;
                const clampedX = Math.max(0, Math.min(1, rawX));
                const clampedY = Math.max(0, Math.min(1, rawY));

                smoothPosRef.current = {
                  x: smoothPosRef.current.x + (clampedX - smoothPosRef.current.x) * SMOOTHING,
                  y: smoothPosRef.current.y + (clampedY - smoothPosRef.current.y) * SMOOTHING,
                };

                setCursorPos({
                  x: smoothPosRef.current.x * 100,
                  y: smoothPosRef.current.y * 100,
                });

                setIsPaused(false);
              }

              // ==========================================
              // MODE: CLICK (2 fingers)
              // ==========================================
              else if (currentMode === "click") {
                // Release drag if was dragging
                if (isDraggingRef.current) {
                  const el = document.elementFromPoint(
                    smoothPosRef.current.x * window.innerWidth,
                    smoothPosRef.current.y * window.innerHeight
                  );
                  if (el) {
                    el.dispatchEvent(new MouseEvent("mouseup", {
                      bubbles: true,
                      clientX: smoothPosRef.current.x * window.innerWidth,
                      clientY: smoothPosRef.current.y * window.innerHeight,
                    }));
                  }
                  isDraggingRef.current = false;
                }

                // Cursor stays frozen, perform click
                if (now - lastClickRef.current > CLICK_COOLDOWN) {
                  lastClickRef.current = now;
                  setClickCount((c) => c + 1);
                  setClickFlash(true);
                  setTimeout(() => setClickFlash(false), 300);

                  // Simulate click at cursor position
                  const clickX = smoothPosRef.current.x * window.innerWidth;
                  const clickY = smoothPosRef.current.y * window.innerHeight;
                  const el = document.elementFromPoint(clickX, clickY);
                  if (el) {
                    el.dispatchEvent(new MouseEvent("click", {
                      bubbles: true,
                      clientX: clickX,
                      clientY: clickY,
                    }));
                  }
                }

                setIsPaused(false);
              }

              // ==========================================
              // MODE: DRAG (3 fingers)
              // ==========================================
              else if (currentMode === "drag") {
                // Start drag if not already dragging
                if (!isDraggingRef.current) {
                  isDraggingRef.current = true;
                  dragStartPosRef.current = { ...smoothPosRef.current };
                  setDragCount((c) => c + 1);

                  const startX = smoothPosRef.current.x * window.innerWidth;
                  const startY = smoothPosRef.current.y * window.innerHeight;
                  const el = document.elementFromPoint(startX, startY);
                  if (el) {
                    el.dispatchEvent(new MouseEvent("mousedown", {
                      bubbles: true,
                      clientX: startX,
                      clientY: startY,
                    }));
                  }
                }

                // Move cursor while dragging
                const sens = sensitivityRef.current;
                const rawX = 0.5 + (indexTip.x - 0.5) * sens;
                const rawY = 0.5 + (indexTip.y - 0.5) * sens;
                const clampedX = Math.max(0, Math.min(1, rawX));
                const clampedY = Math.max(0, Math.min(1, rawY));

                smoothPosRef.current = {
                  x: smoothPosRef.current.x + (clampedX - smoothPosRef.current.x) * SMOOTHING,
                  y: smoothPosRef.current.y + (clampedY - smoothPosRef.current.y) * SMOOTHING,
                };

                setCursorPos({
                  x: smoothPosRef.current.x * 100,
                  y: smoothPosRef.current.y * 100,
                });

                // Dispatch mousemove for drag
                if (now - lastDragRef.current > DRAG_COOLDOWN) {
                  lastDragRef.current = now;
                  const moveX = smoothPosRef.current.x * window.innerWidth;
                  const moveY = smoothPosRef.current.y * window.innerHeight;
                  const el = document.elementFromPoint(moveX, moveY);
                  if (el) {
                    el.dispatchEvent(new MouseEvent("mousemove", {
                      bubbles: true,
                      clientX: moveX,
                      clientY: moveY,
                    }));
                  }
                }

                setIsPaused(false);
              }

              // ==========================================
              // MODE: PAUSED (5 fingers)
              // ==========================================
              else if (currentMode === "paused") {
                // Release drag if was dragging
                if (isDraggingRef.current) {
                  const el = document.elementFromPoint(
                    smoothPosRef.current.x * window.innerWidth,
                    smoothPosRef.current.y * window.innerHeight
                  );
                  if (el) {
                    el.dispatchEvent(new MouseEvent("mouseup", {
                      bubbles: true,
                      clientX: smoothPosRef.current.x * window.innerWidth,
                      clientY: smoothPosRef.current.y * window.innerHeight,
                    }));
                  }
                  isDraggingRef.current = false;
                }

                setIsPaused(true);
              }

              // Draw landmarks
              if (overlayCanvasRef.current) {
                const overlayCtx = overlayCanvasRef.current.getContext("2d");
                if (overlayCtx) {
                  drawLandmarks(
                    overlayCtx,
                    mirroredLandmarks,
                    overlayCanvasRef.current.width,
                    overlayCanvasRef.current.height,
                    currentMode
                  );
                }
              }
            } else {
              setHandDetected(false);
              setFingerCount(0);
              setMode("idle");
              stableModeRef.current = "idle";

              // Release drag if hand lost
              if (isDraggingRef.current) {
                isDraggingRef.current = false;
              }

              if (overlayCanvasRef.current) {
                const overlayCtx = overlayCanvasRef.current.getContext("2d");
                if (overlayCtx) {
                  overlayCtx.clearRect(
                    0, 0,
                    overlayCanvasRef.current.width,
                    overlayCanvasRef.current.height
                  );
                }
              }
            }
          }

          animationRef.current = requestAnimationFrame(detect);
        };

        detect();
      } catch (err) {
        console.error("Error initializing hand tracker:", err);
        setLoadingMessage(
          "Error: Could not access camera. Please allow camera permission and refresh."
        );
      }
    }

    init();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [countFingers, getStableMode, drawLandmarks]);

  // Mode display config
  const getModeConfig = () => {
    switch (mode) {
      case "move":
        return {
          label: "MOVE",
          icon: "☝️",
          color: "#00e5ff",
          bgColor: "rgba(0, 229, 255, 0.12)",
          borderColor: "rgba(0, 229, 255, 0.3)",
          desc: "Moving cursor",
        };
      case "click":
        return {
          label: "CLICK",
          icon: "✌️",
          color: "#ff4466",
          bgColor: "rgba(255, 68, 102, 0.12)",
          borderColor: "rgba(255, 68, 102, 0.3)",
          desc: "Click at position",
        };
      case "drag":
        return {
          label: "DRAG",
          icon: "🤟",
          color: "#ffaa00",
          bgColor: "rgba(255, 170, 0, 0.12)",
          borderColor: "rgba(255, 170, 0, 0.3)",
          desc: "Dragging item",
        };
      case "paused":
        return {
          label: "PAUSED",
          icon: "🖐️",
          color: "#888",
          bgColor: "rgba(120, 120, 140, 0.12)",
          borderColor: "rgba(120, 120, 140, 0.3)",
          desc: "System paused",
        };
      default:
        return {
          label: "IDLE",
          icon: "✋",
          color: "#555",
          bgColor: "rgba(80, 80, 100, 0.08)",
          borderColor: "rgba(80, 80, 100, 0.2)",
          desc: "Show hand to start",
        };
    }
  };

  const modeConfig = getModeConfig();

  // Cursor appearance based on mode
  const getCursorStyle = () => {
    if (isPaused || mode === "idle") {
      return {
        backgroundColor: "rgba(100, 100, 120, 0.4)",
        boxShadow: "0 0 10px rgba(100, 100, 120, 0.2)",
        transform: "translate(-50%, -50%) scale(0.7)",
        opacity: 0.5,
      };
    }
    if (mode === "click" || clickFlash) {
      return {
        backgroundColor: "#ff4466",
        boxShadow: "0 0 30px rgba(255, 68, 102, 0.6), 0 0 60px rgba(255, 68, 102, 0.3)",
        transform: "translate(-50%, -50%) scale(1.6)",
      };
    }
    if (mode === "drag") {
      return {
        backgroundColor: "#ffaa00",
        boxShadow: "0 0 25px rgba(255, 170, 0, 0.5), 0 0 50px rgba(255, 170, 0, 0.2)",
        transform: "translate(-50%, -50%) scale(1.3)",
      };
    }
    // Move mode
    return {
      backgroundColor: "#00e5ff",
      boxShadow: "0 0 20px rgba(0, 229, 255, 0.5), 0 0 40px rgba(0, 229, 255, 0.2)",
      transform: "translate(-50%, -50%) scale(1)",
    };
  };

  return (
    <div className="camera-mouse-app">
      {/* Hidden video element */}
      <video ref={videoRef} style={{ display: "none" }} playsInline muted />

      {/* Loading Screen */}
      {isLoading && (
        <div className="loading-screen">
          <div className="loading-content">
            <div className="loading-icon">
              <svg viewBox="0 0 100 100" className="loading-hand-svg">
                <path
                  d="M50 85 C30 85 20 70 20 55 L20 35 C20 30 25 28 28 30 L28 45 M28 45 L28 20 C28 15 33 13 36 15 L36 42 M36 42 L36 15 C36 10 41 8 44 10 L44 40 M44 40 L44 18 C44 13 49 11 52 13 L52 45 M52 45 L52 30 C52 25 57 23 60 25 L60 55 C60 70 55 85 50 85"
                  fill="none"
                  stroke="url(#handGradient)"
                  strokeWidth="3"
                  strokeLinecap="round"
                  className="loading-hand-path"
                />
                <defs>
                  <linearGradient id="handGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#00c8ff" />
                    <stop offset="100%" stopColor="#a855f7" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <h2 className="loading-title">Camera Mouse</h2>
            <p className="loading-message">{loadingMessage}</p>
            <div className="loading-bar">
              <div className="loading-bar-fill" />
            </div>
          </div>
        </div>
      )}

      {/* Main Interface */}
      {!isLoading && (
        <>
          {/* Paused Overlay */}
          {isPaused && (
            <div className="paused-overlay">
              <div className="paused-content">
                <div className="paused-icon">🖐️</div>
                <h2 className="paused-title">PAUSED</h2>
                <p className="paused-desc">Show fewer fingers to resume</p>
              </div>
            </div>
          )}

          {/* Floating Cursor */}
          <div
            className="floating-cursor"
            style={{
              left: `${cursorPos.x}%`,
              top: `${cursorPos.y}%`,
              ...getCursorStyle(),
            }}
          >
            {clickFlash && <div className="cursor-ripple" />}
            {mode === "drag" && isDraggingRef.current && (
              <div className="cursor-drag-indicator">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M5 9l7-7 7 7M5 15l7 7 7-7" />
                </svg>
              </div>
            )}
          </div>

          {/* Mode Indicator - Top Center */}
          {handDetected && (
            <div
              className="mode-indicator"
              style={{
                backgroundColor: modeConfig.bgColor,
                borderColor: modeConfig.borderColor,
                color: modeConfig.color,
              }}
            >
              <span className="mode-indicator-icon">{modeConfig.icon}</span>
              <span className="mode-indicator-label">{modeConfig.label}</span>
              <span className="mode-indicator-fingers">{fingerCount}F</span>
            </div>
          )}

          {/* Camera Preview */}
          <div
            className="camera-preview"
            style={{ opacity: showCamera ? 1 : 0, pointerEvents: showCamera ? "auto" : "none" }}
          >
            <div className="camera-preview-header">
              <div className="camera-dot" />
              <span>LIVE</span>
              <span className="camera-fps">{fps} FPS</span>
            </div>
            <canvas ref={canvasRef} className="camera-canvas" />
            {showLandmarks && (
              <canvas ref={overlayCanvasRef} className="overlay-canvas" />
            )}
            {!handDetected && isTracking && (
              <div className="no-hand-overlay">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8" />
                  <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
                </svg>
                <p>Show your hand</p>
              </div>
            )}
          </div>

          {/* Control Panel */}
          <div className="control-panel">
            <div className="panel-header">
              <h1 className="panel-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="url(#titleGrad)" strokeWidth="2">
                  <defs>
                    <linearGradient id="titleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#00c8ff" />
                      <stop offset="100%" stopColor="#a855f7" />
                    </linearGradient>
                  </defs>
                  <path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8" />
                  <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
                </svg>
                Camera Mouse
              </h1>
              <div
                className="status-badge"
                style={{ color: handDetected ? "#44ff88" : "#ff8844" }}
              >
                <div
                  className="status-dot"
                  style={{ backgroundColor: handDetected ? "#44ff88" : "#ff8844" }}
                />
                {handDetected ? "Active" : "No Hand"}
              </div>
            </div>

            {/* Current Mode Display */}
            <div
              className="current-mode-card"
              style={{
                background: modeConfig.bgColor,
                borderColor: modeConfig.borderColor,
              }}
            >
              <div className="current-mode-top">
                <span className="current-mode-icon">{modeConfig.icon}</span>
                <div className="current-mode-info">
                  <span className="current-mode-label" style={{ color: modeConfig.color }}>
                    {modeConfig.label}
                  </span>
                  <span className="current-mode-desc">{modeConfig.desc}</span>
                </div>
                <span className="current-mode-fingers" style={{ color: modeConfig.color }}>
                  {fingerCount}
                </span>
              </div>
            </div>

            {/* Stats */}
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-label">Clicks</span>
                <span className="stat-value">{clickCount}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Drags</span>
                <span className="stat-value">{dragCount}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Position</span>
                <span className="stat-value">
                  {Math.round(cursorPos.x)}%, {Math.round(cursorPos.y)}%
                </span>
              </div>
              <div className="stat-card">
                <span className="stat-label">FPS</span>
                <span className="stat-value">{fps}</span>
              </div>
            </div>

            {/* Controls */}
            <div className="controls-section">
              <div className="control-row">
                <label className="control-label">Sensitivity</label>
                <div className="slider-container">
                  <input
                    type="range"
                    min="1"
                    max="3"
                    step="0.1"
                    value={sensitivity}
                    onChange={(e) => setSensitivity(parseFloat(e.target.value))}
                    className="slider"
                  />
                  <span className="slider-value">{sensitivity.toFixed(1)}x</span>
                </div>
              </div>

              <div className="control-row">
                <label className="control-label">Landmarks</label>
                <button
                  className={`toggle-btn ${showLandmarks ? "active" : ""}`}
                  onClick={() => setShowLandmarks(!showLandmarks)}
                >
                  <div className="toggle-thumb" />
                </button>
              </div>

              <div className="control-row">
                <label className="control-label">Camera</label>
                <button
                  className={`toggle-btn ${showCamera ? "active" : ""}`}
                  onClick={() => setShowCamera(!showCamera)}
                >
                  <div className="toggle-thumb" />
                </button>
              </div>
            </div>

            {/* Gesture Guide */}
            <div className="instructions">
              <h3 className="instructions-title">Gesture Guide</h3>
              <div className={`instruction-item ${mode === "move" ? "instruction-active" : ""}`}>
                <span className="instruction-icon">☝️</span>
                <span className="instruction-text">1 finger</span>
                <span className="instruction-action" style={{ color: "#00e5ff" }}>Move</span>
              </div>
              <div className={`instruction-item ${mode === "click" ? "instruction-active" : ""}`}>
                <span className="instruction-icon">✌️</span>
                <span className="instruction-text">2 fingers</span>
                <span className="instruction-action" style={{ color: "#ff4466" }}>Click</span>
              </div>
              <div className={`instruction-item ${mode === "drag" ? "instruction-active" : ""}`}>
                <span className="instruction-icon">🤟</span>
                <span className="instruction-text">3 fingers</span>
                <span className="instruction-action" style={{ color: "#ffaa00" }}>Drag</span>
              </div>
              <div className={`instruction-item ${mode === "paused" ? "instruction-active" : ""}`}>
                <span className="instruction-icon">🖐️</span>
                <span className="instruction-text">5 fingers</span>
                <span className="instruction-action" style={{ color: "#888" }}>Pause</span>
              </div>
            </div>
          </div>

          {/* Click Flash Indicator */}
          {clickFlash && (
            <div className="click-indicator">CLICK!</div>
          )}
        </>
      )}
    </div>
  );
}
