import cv2
import sys


def test_placeholder():
    """Visual camera tests require manual hardware run; see __main__ block below."""
    pass


def run_camera_test():
    print("Testing camera access...")

    # Try to open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open camera at index 0")
        print("Make sure:")
        print("  1. Your camera is connected")
        print("  2. No other application is using the camera")
        print("  3. Camera permissions are granted")
        sys.exit(1)

    print("✓ Camera opened successfully")

    # Try to read a frame
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read frame from camera")
        cap.release()
        sys.exit(1)

    print(f"✓ Frame captured: {frame.shape}")

    # Try to create a window
    try:
        cv2.namedWindow("Test Camera", cv2.WINDOW_AUTOSIZE)
        print("✓ Window created")
        
        print("\nDisplaying camera feed...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Lost camera feed")
                break
                
            cv2.imshow("Test Camera", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User quit")
                break
                
    except Exception as e:
        print(f"ERROR: {e}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera test completed")


if __name__ == "__main__":
    run_camera_test()
