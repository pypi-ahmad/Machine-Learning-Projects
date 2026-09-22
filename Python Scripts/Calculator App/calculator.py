import ast
import math
from tkinter import Tk, END, Entry, N, E, S, W, Button, messagebox
from tkinter import font
from functools import partial


def get_input(entry, argu):
    entry.insert(END, argu)


def backspace(entry):
    input_len = len(entry.get())
    if input_len:
        entry.delete(input_len - 1)


def clear(entry):
    entry.delete(0, END)


def evaluate_expression(expression):
    """Evaluate a calculator expression containing only numbers and arithmetic."""
    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool) or not math.isfinite(node.value):
                raise ValueError("Enter finite numbers only.")
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
        raise ValueError("Use numbers and +, -, *, /, or ^ only.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Enter a valid expression.") from error

    result = evaluate(tree.body)
    if not math.isfinite(result):
        raise ValueError("The result must be finite.")
    return result


def calc(entry):
    try:
        output = str(evaluate_expression(entry.get().strip()))
    except ZeroDivisionError:
        popupmsg("Cannot divide by zero. Enter valid values.")
        return
    except ValueError as error:
        popupmsg(str(error))
        return

    clear(entry)
    entry.insert(END, output)


def popupmsg(message):
    messagebox.showerror("Calculator", message)


def cal():
    root = Tk()
    root.title("Calc")
    root.resizable(0, 0)

    entry_font = font.Font(size=15)
    entry = Entry(root, justify="right", font=entry_font)
    entry.grid(row=0, column=0, columnspan=4,
               sticky=N + W + S + E, padx=5, pady=5)

    cal_button_bg = '#FF6600'
    num_button_bg = '#4B4B4B'
    other_button_bg = '#DDDDDD'
    text_fg = '#FFFFFF'
    button_active_bg = '#C0C0C0'

    num_button = partial(Button, root, fg=text_fg, bg=num_button_bg,
                         padx=10, pady=3, activebackground=button_active_bg)
    cal_button = partial(Button, root, fg=text_fg, bg=cal_button_bg,
                         padx=10, pady=3, activebackground=button_active_bg)

    button7 = num_button(text='7', bg=num_button_bg,
                         command=lambda: get_input(entry, '7'))
    button7.grid(row=2, column=0, pady=5)

    button8 = num_button(text='8', command=lambda: get_input(entry, '8'))
    button8.grid(row=2, column=1, pady=5)

    button9 = num_button(text='9', command=lambda: get_input(entry, '9'))
    button9.grid(row=2, column=2, pady=5)

    button10 = cal_button(text='+', command=lambda: get_input(entry, '+'))
    button10.grid(row=4, column=3, pady=5)

    button4 = num_button(text='4', command=lambda: get_input(entry, '4'))
    button4.grid(row=3, column=0, pady=5)

    button5 = num_button(text='5', command=lambda: get_input(entry, '5'))
    button5.grid(row=3, column=1, pady=5)

    button6 = num_button(text='6', command=lambda: get_input(entry, '6'))
    button6.grid(row=3, column=2, pady=5)

    button11 = cal_button(text='-', command=lambda: get_input(entry, '-'))
    button11.grid(row=3, column=3, pady=5)

    button1 = num_button(text='1', command=lambda: get_input(entry, '1'))
    button1.grid(row=4, column=0, pady=5)

    button2 = num_button(text='2', command=lambda: get_input(entry, '2'))
    button2.grid(row=4, column=1, pady=5)

    button3 = num_button(text='3', command=lambda: get_input(entry, '3'))
    button3.grid(row=4, column=2, pady=5)

    button12 = cal_button(text='*', command=lambda: get_input(entry, '*'))
    button12.grid(row=2, column=3, pady=5)

    button0 = num_button(text='0', command=lambda: get_input(entry, '0'))
    #button0.grid(row=5, column=0, columnspan=2, padx=3, pady=5, sticky=N + S + E + W)
    button0.grid(row=5, column=0,  pady=5)

    button13 = num_button(text='.', command=lambda: get_input(entry, '.'))
    button13.grid(row=5, column=1, pady=5)

    button14 = Button(root, text='/', fg=text_fg, bg=cal_button_bg, padx=10, pady=3,
                      command=lambda: get_input(entry, '/'))
    button14.grid(row=1, column=3, pady=5)

    button15 = Button(root, text='<-', bg=other_button_bg, padx=10, pady=3,
                      command=lambda: backspace(entry), activebackground=button_active_bg)
    button15.grid(row=1, column=0, columnspan=2,
                  padx=3, pady=5, sticky=N + S + E + W)

    button16 = Button(root, text='C', bg=other_button_bg, padx=10, pady=3,
                      command=lambda: clear(entry), activebackground=button_active_bg)
    button16.grid(row=1, column=2, pady=5)

    button17 = Button(root, text='=', fg=text_fg, bg=cal_button_bg, padx=10, pady=3,
                      command=lambda: calc(entry), activebackground=button_active_bg)
    button17.grid(row=5, column=3, pady=5)

    button18 = Button(root, text='^', fg=text_fg, bg=cal_button_bg, padx=10, pady=3,
                      command=lambda: get_input(entry, '**'))
    button18.grid(row=5, column=2, pady=5)

    root.mainloop()


if __name__ == '__main__':
    cal()
