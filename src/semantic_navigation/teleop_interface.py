"""
User command interface for the navigation system.

Provides both a terminal-based interface and a simple tkinter GUI
for sending navigation commands and viewing status.
"""

import sys
import threading
import time


class TerminalInterface:
    """Simple terminal-based command interface."""

    HELP_TEXT = """
=== Semantic Navigation Command Interface ===

Navigation commands:
  Go to the <object>         - Navigate to an object
  Find the <color> <object>  - Find an object with attribute
  Navigate to <obj> near <ref> - Navigate using spatial reference

Control commands:
  start exploration  - Begin autonomous exploration
  stop exploration   - Stop exploration
  show map           - Display semantic map contents
  help               - Show this help
  quit               - Exit

Examples:
  Go to the blue vase
  Find the red chair in the bedroom
  Navigate to the table near the sofa
"""

    def __init__(self, publish_fn=None):
        """
        Args:
            publish_fn: Callable that takes a string command and publishes it.
                       For ROS2, this publishes to /user_command topic.
                       For standalone, this calls the mission controller directly.
        """
        self.publish_fn = publish_fn
        self.running = False
        self.status_log: list[str] = []

    def on_status(self, text: str):
        """Called when a status update is received."""
        self.status_log.append(text)
        print(f"\n[STATUS] {text}")

    def run(self):
        """Run the terminal interface (blocking)."""
        print(self.HELP_TEXT)
        self.running = True

        while self.running:
            try:
                cmd = input("\nCommand> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not cmd:
                continue
            if cmd.lower() == "quit":
                break
            if cmd.lower() == "help":
                print(self.HELP_TEXT)
                continue

            if self.publish_fn:
                self.publish_fn(cmd)
            else:
                print(f"[SENT] {cmd}")

        self.running = False


class TkinterInterface:
    """Simple tkinter GUI for commands and status."""

    def __init__(self, publish_fn=None):
        self.publish_fn = publish_fn
        self.root = None

    def on_status(self, text: str):
        """Called when a status update is received."""
        if self.root and hasattr(self, "status_text"):
            self.status_text.insert("end", f"{text}\n")
            self.status_text.see("end")

    def run(self):
        """Run the tkinter GUI (blocking)."""
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title("Semantic Navigation - Command Interface")
        self.root.geometry("600x500")

        # Title
        tk.Label(
            self.root, text="Vision-Language Navigation", font=("Arial", 16, "bold")
        ).pack(pady=10)

        # Command entry
        cmd_frame = tk.Frame(self.root)
        cmd_frame.pack(fill="x", padx=10)
        tk.Label(cmd_frame, text="Command:").pack(side="left")
        self.cmd_entry = tk.Entry(cmd_frame, font=("Arial", 12))
        self.cmd_entry.pack(side="left", fill="x", expand=True, padx=5)
        tk.Button(cmd_frame, text="Send", command=self._send_command).pack(side="right")

        # Bind Enter key
        self.cmd_entry.bind("<Return>", lambda e: self._send_command())

        # Quick buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=5)
        for text in ["Start Exploration", "Stop Exploration", "Show Map"]:
            tk.Button(
                btn_frame, text=text,
                command=lambda t=text: self._send(t)
            ).pack(side="left", padx=2)

        # Status log
        tk.Label(self.root, text="Status:", font=("Arial", 11, "bold")).pack(
            anchor="w", padx=10
        )
        self.status_text = tk.Text(self.root, height=15, font=("Courier", 10))
        self.status_text.pack(fill="both", expand=True, padx=10, pady=5)

        # Example commands
        tk.Label(
            self.root,
            text='Examples: "Go to the blue vase" | "Find the red chair in the bedroom"',
            font=("Arial", 9), fg="gray",
        ).pack(pady=5)

        self.root.mainloop()

    def _send_command(self):
        cmd = self.cmd_entry.get().strip()
        if cmd:
            self._send(cmd)
            self.cmd_entry.delete(0, "end")

    def _send(self, text: str):
        self.status_text.insert("end", f"> {text}\n")
        self.status_text.see("end")
        if self.publish_fn:
            self.publish_fn(text)


# ---- ROS2 node version ----
def create_ros_interface():
    """Create a ROS2 teleop interface node."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class TeleopNode(Node):
        def __init__(self):
            super().__init__("teleop_interface")
            self.cmd_pub = self.create_publisher(String, "/user_command", 10)
            self.status_sub = self.create_subscription(
                String, "/navigation_status", self.status_cb, 10
            )
            self.interface = TerminalInterface(publish_fn=self.publish_cmd)

        def publish_cmd(self, text):
            msg = String()
            msg.data = text
            self.cmd_pub.publish(msg)

        def status_cb(self, msg):
            self.interface.on_status(msg.data)

        def run(self):
            # Run ROS spin in background thread
            spin_thread = threading.Thread(
                target=rclpy.spin, args=(self,), daemon=True
            )
            spin_thread.start()
            self.interface.run()

    rclpy.init()
    node = TeleopNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


# ---- Standalone test ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Navigation command interface")
    parser.add_argument(
        "--gui", action="store_true", help="Use tkinter GUI instead of terminal"
    )
    parser.add_argument(
        "--ros", action="store_true", help="Run as ROS2 node"
    )
    args = parser.parse_args()

    if args.ros:
        create_ros_interface()
    elif args.gui:
        ui = TkinterInterface(publish_fn=lambda cmd: print(f"[SEND] {cmd}"))
        ui.run()
    else:
        ui = TerminalInterface(publish_fn=lambda cmd: print(f"[SEND] {cmd}"))
        ui.run()
