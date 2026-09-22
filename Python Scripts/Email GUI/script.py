"""Compose and send a Gmail SMTP email from a Tkinter window."""

import smtplib
import ssl
import tkinter as tk
from email.message import EmailMessage
from tkinter import messagebox

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def build_message(sender: str, recipient: str, subject: str, body: str) -> EmailMessage:
    """Create a plain-text email with the supplied headers and content."""
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    return message


def send_email(sender: str, app_password: str, recipient: str, subject: str, body: str) -> None:
    """Send one message through Gmail SMTP using STARTTLS."""
    message = build_message(sender, recipient, subject, body)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(sender, app_password)
        server.send_message(message)


def main() -> None:
    """Create the email composer window."""
    gui = tk.Tk()
    gui.geometry("500x500")
    gui.title("Email sender")
    gui.configure(background="light blue")

    sender_var = tk.StringVar()
    password_var = tk.StringVar()
    recipient_var = tk.StringVar()
    subject_var = tk.StringVar()

    tk.Label(gui, text="Email sender", background="yellow", foreground="black", font=("Segoe UI", 16, "bold"), height=2).pack(fill=tk.X)
    form = tk.Frame(gui, background="light blue")
    form.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

    def add_entry(label: str, variable: tk.StringVar, row: int, secret: bool = False) -> None:
        tk.Label(form, text=label, background="light blue").grid(row=row, column=0, sticky="w", pady=(0, 4))
        tk.Entry(form, textvariable=variable, width=45, show="*" if secret else "").grid(row=row + 1, column=0, sticky="ew", pady=(0, 10))

    add_entry("Sender Gmail address", sender_var, 0)
    add_entry("Gmail app password", password_var, 2, secret=True)
    add_entry("Recipient email address", recipient_var, 4)
    add_entry("Subject", subject_var, 6)
    tk.Label(form, text="Message", background="light blue").grid(row=8, column=0, sticky="w", pady=(0, 4))
    body_text = tk.Text(form, width=45, height=8, wrap=tk.WORD)
    body_text.grid(row=9, column=0, sticky="nsew")
    form.columnconfigure(0, weight=1)
    form.rowconfigure(9, weight=1)

    def send_message() -> None:
        sender = sender_var.get().strip()
        recipient = recipient_var.get().strip()
        subject = subject_var.get().strip()
        body = body_text.get("1.0", "end-1c").strip()
        app_password = password_var.get()
        if not all((sender, recipient, subject, body, app_password)):
            messagebox.showerror("Missing information", "Complete every field before sending.")
            return
        if not messagebox.askyesno("Confirm email", f"Send this message to {recipient}?"):
            return
        try:
            send_email(sender, app_password, recipient, subject, body)
        except (OSError, smtplib.SMTPException) as error:
            messagebox.showerror("Email not sent", str(error))
            return
        password_var.set("")
        subject_var.set("")
        body_text.delete("1.0", tk.END)
        messagebox.showinfo("Email sent", "The message was accepted by the SMTP server.")

    tk.Button(form, text="Send message", command=send_message, width=30, height=2, background="grey").grid(row=10, column=0, pady=15)
    gui.mainloop()


if __name__ == "__main__":
    main()
