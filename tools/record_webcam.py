import cv2
import os
import time
from pathlib import Path
import argparse

def record_reps(sign_name, num_reps=30, rep_duration=2.5, camera_id=0):
    raw_dir = Path("dataset/raw") / sign_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 120:
        fps = 30.0
        
    print(f"Recording '{sign_name}'. Resolution: {width}x{height} @ {fps}fps")
    print(f"Will record {num_reps} reps, each lasting {rep_duration} seconds.")
    print("Press 'SPACE' to start the next rep.")
    print("Press 'q' at any time to quit completely.")
    
    rep = 1
    while rep <= num_reps:
        print(f"\n--- Ready for Rep {rep}/{num_reps} ---")
        print("Press SPACE to start recording this rep (or 'q' to quit)...")
        
        # Wait for space or q
        start_recording = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Draw preview text
            preview = frame.copy()
            cv2.putText(preview, f"Sign: {sign_name} | Rep: {rep}/{num_reps}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(preview, "Press SPACE to start", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Webcam Record", preview)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                start_recording = True
                break
            elif key == ord('q'):
                print("Quitting...")
                cap.release()
                cv2.destroyAllWindows()
                return

        if not start_recording:
            break
            
        # Countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            preview = frame.copy()
            cv2.putText(preview, f"Starting in {i}...", (width//2 - 100, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imshow("Webcam Record", preview)
            cv2.waitKey(1000)
            
        print(f"Recording rep {rep}...")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = raw_dir / f"rep_{rep:03d}.mp4"
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        
        start_time = time.time()
        frames_recorded = 0
        
        while (time.time() - start_time) < rep_duration:
            ret, frame = cap.read()
            if not ret:
                break
                
            out.write(frame)
            frames_recorded += 1
            
            # Show recording status
            preview = frame.copy()
            time_left = rep_duration - (time.time() - start_time)
            cv2.putText(preview, f"RECORDING - {time_left:.1f}s left", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(preview, (width - 30, 30), 10, (0, 0, 255), -1)
            cv2.imshow("Webcam Record", preview)
            
            cv2.waitKey(1)
            
        out.release()
        print(f"Saved {out_path} ({frames_recorded} frames)")
        rep += 1
        
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished recording {num_reps} reps for '{sign_name}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record webcam footage for sign language dataset.")
    parser.add_argument("--sign", type=str, required=True, help="Name of the sign to record")
    parser.add_argument("--reps", type=int, default=30, help="Number of repetitions to record (default: 30)")
    parser.add_argument("--duration", type=float, default=2.5, help="Duration of each repetition in seconds (default: 2.5)")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    
    args = parser.parse_args()
    record_reps(args.sign, args.reps, args.duration, args.camera)
