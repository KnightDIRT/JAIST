# realtime_pointcloud_fps_clean.py
import open3d as o3d
import numpy as np
import time

def generate_random_point_cloud(num_points=5000):
    """Generate a random point cloud within a unit cube."""
    return np.random.rand(num_points, 3)

def main():
    num_points = 5000
    points = generate_random_point_cloud(num_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Real-Time Point Cloud", width=960, height=720)
    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.background_color = np.array([0.0, 0.0, 0.0])

    print("[INFO] Starting real-time point cloud simulation (Ctrl+C to stop)")

    # FPS tracking
    fps_update_interval = 0.5  # seconds between FPS reports
    last_fps_time = time.time()
    frames_since_last = 0

    try:
        while True:
            # simple motion model: tiny gaussian jitter
            points += np.random.normal(scale=0.001, size=points.shape)
            pcd.points = o3d.utility.Vector3dVector(points)

            # update Open3D visualizer
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # FPS accounting
            frames_since_last += 1
            now = time.time()
            elapsed = now - last_fps_time
            if elapsed >= fps_update_interval:
                fps = frames_since_last / elapsed
                # print to console
                print(f"[FPS] {fps:.2f}")

                # try to update the window title if possible (non-fatal)
                try:
                    # some Open3D builds expose a window object with set_window_title / title attribute.
                    # We attempt a couple of common names; if they don't exist we ignore the exception.
                    w = getattr(vis, "get_window", None)
                    if callable(w):
                        window_obj = vis.get_window()
                        # try common setters (these may or may not exist)
                        if hasattr(window_obj, "set_window_title"):
                            window_obj.set_window_title(f"Real-Time Point Cloud — FPS: {fps:.2f}")
                        elif hasattr(window_obj, "set_title"):
                            window_obj.set_title(f"Real-Time Point Cloud — FPS: {fps:.2f}")
                        elif hasattr(window_obj, "title"):
                            try:
                                window_obj.title = f"Real-Time Point Cloud — FPS: {fps:.2f}"
                            except Exception:
                                pass
                except Exception:
                    # silent fallback; print already provides FPS
                    pass

                # reset counters
                last_fps_time = now
                frames_since_last = 0

            # small sleep to be kind to CPU (tune as needed)
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        vis.destroy_window()

if __name__ == "__main__":
    main()
