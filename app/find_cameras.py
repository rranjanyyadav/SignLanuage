
import cv2
import time

BACKENDS = [
    ("DirectShow", cv2.CAP_DSHOW),
    ("Media Foundation", cv2.CAP_MSMF),
]

print("Sign Lang - Camo Camera Diagnostic")
print("Keep Camo Studio open with live iPhone video.")
print("Press N to test the next camera/backend.")
print("Press Q to quit.")

for backend_name, backend_id in BACKENDS:
    for index in range(6):
        print(f"\nTesting camera {index} with {backend_name}...")

        camera = cv2.VideoCapture(index, backend_id)

        if not camera.isOpened():
            print("  Could not open.")
            camera.release()
            continue

        # Let the camera initialize and discard initial frames.
        frame = None

        for _ in range(15):
            success, candidate = camera.read()
            if success:
                frame = candidate
            time.sleep(0.05)

        if frame is None:
            print("  No frames received.")
            camera.release()
            continue

        mean_brightness = frame.mean()
        variation = frame.std()

        print(
            f"  Frame brightness: {mean_brightness:.1f}; "
            f"variation: {variation:.1f}"
        )

        window_name = f"{backend_name} - Camera {index}"

        while True:
            success, frame = camera.read()

            if not success:
                print("  Frame read failed.")
                break

            cv2.putText(
                frame,
                f"{backend_name} | Camera {index}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("n"):
                break

            if key == ord("q"):
                camera.release()
                cv2.destroyAllWindows()
                raise SystemExit

        camera.release()
        cv2.destroyAllWindows()

print("\nDiagnostic finished.")
