"""Local Chat App — Tkinter desktop app.

Peer-to-peer chat over a local network using Python sockets.
Run as server on one machine and client on another (or same machine).

Usage:
    python main.py
"""

import socket
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


class ChatApp(tk.Tk):
    DEFAULT_PORT = 9090
    BUFFER_SIZE  = 4096

    def __init__(self):
        super().__init__()
        self.title("Local Chat")
        self.geometry("700x560")
        self.configure(bg="#1e1e2e")

        self._sock      = None
        self._conn      = None
        self._mode      = None   # "server" or "client"
        self._connected = False
        self._username  = "Me"

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Connection bar
        conn_fr = tk.Frame(self, bg="#181825")
        conn_fr.pack(fill="x", padx=0, pady=0)

        tk.Label(conn_fr, text="Name:", bg="#181825", fg="#888").pack(side="left", padx=(8, 2))
        self._name_entry = tk.Entry(conn_fr, bg="#313244", fg="#cdd6f4", width=10,
                                     insertbackground="#cba6f7", font=("Consolas", 10),
                                     relief="flat")
        self._name_entry.insert(0, "User1")
        self._name_entry.pack(side="left", padx=4)

        tk.Label(conn_fr, text="Host:", bg="#181825", fg="#888").pack(side="left", padx=(8, 2))
        self._host_entry = tk.Entry(conn_fr, bg="#313244", fg="#cdd6f4", width=14,
                                     insertbackground="#cba6f7", font=("Consolas", 10),
                                     relief="flat")
        self._host_entry.insert(0, "127.0.0.1")
        self._host_entry.pack(side="left", padx=4)

        tk.Label(conn_fr, text="Port:", bg="#181825", fg="#888").pack(side="left", padx=(4, 2))
        self._port_entry = tk.Entry(conn_fr, bg="#313244", fg="#cdd6f4", width=6,
                                     insertbackground="#cba6f7", font=("Consolas", 10),
                                     relief="flat")
        self._port_entry.insert(0, str(self.DEFAULT_PORT))
        self._port_entry.pack(side="left", padx=4)

        self._server_btn = tk.Button(conn_fr, text="Start Server",
                                      command=self._start_server,
                                      bg="#a6e3a1", fg="#1e1e2e", relief="flat",
                                      font=("Consolas", 9, "bold"))
        self._server_btn.pack(side="left", padx=4, pady=6)
        self._client_btn = tk.Button(conn_fr, text="Connect",
                                      command=self._connect_client,
                                      bg="#89b4fa", fg="#1e1e2e", relief="flat",
                                      font=("Consolas", 9, "bold"))
        self._client_btn.pack(side="left", padx=2, pady=6)
        self._disc_btn = tk.Button(conn_fr, text="Disconnect",
                                    command=self._disconnect,
                                    bg="#f38ba8", fg="#1e1e2e", relief="flat",
                                    font=("Consolas", 9), state="disabled")
        self._disc_btn.pack(side="left", padx=2, pady=6)

        self._status_var = tk.StringVar(value="Not connected")
        tk.Label(conn_fr, textvariable=self._status_var, bg="#181825", fg="#888",
                 font=("Consolas", 9)).pack(side="right", padx=8)

        # Chat area
        self._chat = tk.Text(self, bg="#313244", fg="#cdd6f4", font=("Consolas", 11),
                              relief="flat", state="disabled", wrap="word",
                              padx=8, pady=8)
        self._chat.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self._chat.tag_configure("system",  foreground="#888",      font=("Consolas", 9, "italic"))
        self._chat.tag_configure("me",      foreground="#cba6f7",   font=("Consolas", 11, "bold"))
        self._chat.tag_configure("them",    foreground="#89b4fa",   font=("Consolas", 11, "bold"))
        self._chat.tag_configure("msg",     foreground="#cdd6f4")
        self._chat.tag_configure("time",    foreground="#555",      font=("Consolas", 9))

        # Input
        input_fr = tk.Frame(self, bg="#1e1e2e")
        input_fr.pack(fill="x", padx=8, pady=(0, 8))
        self._msg_entry = tk.Entry(input_fr, bg="#313244", fg="#cdd6f4",
                                    insertbackground="#cba6f7", font=("Consolas", 12),
                                    relief="flat")
        self._msg_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._msg_entry.bind("<Return>", lambda _: self._send())
        tk.Button(input_fr, text="Send ▶", command=self._send,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 11, "bold")).pack(side="right")

    # ── Network ───────────────────────────────────────────────────────────────

    def _start_server(self):
        if self._connected:
            return
        port = self._get_port()
        if port is None:
            return
        self._username = self._name_entry.get().strip() or "User1"
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("0.0.0.0", port))
            self._sock.listen(1)
            self._mode = "server"
            self._append_system(f"Server listening on port {port}. Waiting for connection...")
            self._status_var.set(f"Listening on :{port}")
            threading.Thread(target=self._accept_connection, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _accept_connection(self):
        try:
            self._conn, addr = self._sock.accept()
            self._connected = True
            host_str = f"{addr[0]}:{addr[1]}"
            self.after(0, lambda: self._on_connected(f"Client connected from {host_str}"))
            threading.Thread(target=self._receive_loop, daemon=True).start()
        except Exception:
            pass

    def _connect_client(self):
        if self._connected:
            return
        host = self._host_entry.get().strip()
        port = self._get_port()
        if not host or port is None:
            return
        self._username = self._name_entry.get().strip() or "User1"
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((host, port))
            self._conn  = self._sock
            self._mode  = "client"
            self._connected = True
            self.after(0, lambda: self._on_connected(f"Connected to {host}:{port}"))
            threading.Thread(target=self._receive_loop, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))

    def _on_connected(self, msg: str):
        self._append_system(msg)
        self._status_var.set("Connected")
        self._server_btn.config(state="disabled")
        self._client_btn.config(state="disabled")
        self._disc_btn.config(state="normal")
        # Announce username
        self._raw_send(f"__JOIN__{self._username}")

    def _receive_loop(self):
        while self._connected:
            try:
                data = self._conn.recv(self.BUFFER_SIZE)
                if not data:
                    break
                text = data.decode("utf-8", errors="replace")
                self.after(0, lambda t=text: self._handle_incoming(t))
            except Exception:
                break
        self.after(0, self._on_disconnected)

    def _handle_incoming(self, text: str):
        if text.startswith("__JOIN__"):
            peer = text[8:]
            self._append_system(f"  {peer} joined the chat.")
        elif text.startswith("__MSG__"):
            parts = text[7:].split("__", 1)
            sender = parts[0] if len(parts) == 2 else "Peer"
            msg    = parts[1] if len(parts) == 2 else text[7:]
            self._append_message(sender, msg, "them")
        elif text.startswith("__LEAVE__"):
            peer = text[9:]
            self._append_system(f"  {peer} left the chat.")

    def _send(self):
        if not self._connected:
            self._append_system("Not connected. Start or join a server first.")
            return
        msg = self._msg_entry.get().strip()
        if not msg:
            return
        self._msg_entry.delete(0, "end")
        self._raw_send(f"__MSG__{self._username}__{msg}")
        self._append_message(self._username, msg, "me")

    def _raw_send(self, text: str):
        try:
            self._conn.sendall(text.encode("utf-8"))
        except Exception as e:
            self._append_system(f"Send error: {e}")

    def _disconnect(self):
        if self._connected:
            try:
                self._raw_send(f"__LEAVE__{self._username}")
            except Exception:
                pass
        self._on_disconnected()

    def _on_disconnected(self):
        self._connected = False
        try:
            if self._conn:
                self._conn.close()
            if self._mode == "server" and self._sock and self._conn is not self._sock:
                self._sock.close()
        except Exception:
            pass
        self._conn = self._sock = None
        self._mode = None
        self._append_system("Disconnected.")
        self._status_var.set("Disconnected")
        self._server_btn.config(state="normal")
        self._client_btn.config(state="normal")
        self._disc_btn.config(state="disabled")

    def _get_port(self) -> int | None:
        try:
            return int(self._port_entry.get().strip())
        except ValueError:
            messagebox.showerror("Port", "Enter a valid port number.")
            return None

    # ── Chat display ──────────────────────────────────────────────────────────

    def _append_message(self, sender: str, msg: str, tag: str):
        self._chat.config(state="normal")
        self._chat.insert("end", f"[{timestamp()}] ", "time")
        self._chat.insert("end", f"{sender}: ", tag)
        self._chat.insert("end", msg + "\n", "msg")
        self._chat.see("end")
        self._chat.config(state="disabled")

    def _append_system(self, msg: str):
        self._chat.config(state="normal")
        self._chat.insert("end", f"{msg}\n", "system")
        self._chat.see("end")
        self._chat.config(state="disabled")

    def _on_close(self):
        self._disconnect()
        self.destroy()


if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
