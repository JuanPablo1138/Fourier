import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import scipy.io.wavfile as wav
from tkinter import filedialog

# === Funciones base ===
def funcion_lineal(x): return x
def funcion_onda_triangular(x): return 2 * (np.abs((x / np.pi) % 2 - 1)) - 1
def funcion_onda_sierra(x): return 2 * ((x / np.pi) % 1) - 1
def funcion_onda_cuadrada(x): return np.sign(np.sin(x))

funciones = [
    ("Función Lineal (f(x) = x)", funcion_lineal),
    ("Onda Triangular", funcion_onda_triangular),
    ("Onda Sierra", funcion_onda_sierra),
    ("Onda Cuadrada", funcion_onda_cuadrada)
]

# === Cálculo de coeficientes ===
def calcular_coeficientes(f, T, N):
    a0 = (2 / T) * quad(f, -T / 2, T / 2)[0]
    an = []
    bn = []
    for n in range(1, N + 1):
        an.append((2 / T) * quad(lambda x: f(x) * np.cos(2 * np.pi * n * x / T), -T / 2, T / 2)[0])
        bn.append((2 / T) * quad(lambda x: f(x) * np.sin(2 * np.pi * n * x / T), -T / 2, T / 2)[0])
    return a0, an, bn

def fourier_series(x, a0, an, bn, N, T):
    suma = a0 / 2
    for n in range(1, N + 1):
        suma += an[n - 1] * np.cos(2 * np.pi * n * x / T) + bn[n - 1] * np.sin(2 * np.pi * n * x / T)
    return suma

# === Gráfica ===
def graficar_fourier(funcion, nombre, a0, an, bn, N, T, muestras, x_min, x_max):
    x_fourier = np.linspace(x_min * 3, x_max * 3, muestras * 3)
    x_funcion = np.linspace(x_min, x_max, muestras)
    y_original = np.vectorize(funcion)(x_funcion)
    y_fourier = np.vectorize(lambda x: fourier_series(x, a0, an, bn, N, T))(x_fourier)

    plt.figure(figsize=(12, 8))
    plt.plot(x_funcion, y_original, label="Función Original", color="red")
    plt.plot(x_fourier, y_fourier, label=f"Serie de Fourier (N={N})", color="blue")
    plt.title(f"Aproximación de la Serie de Fourier para {nombre}", fontsize=16)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("f(x)", fontsize=14)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

def mostrar_coeficientes(a0, an, bn):
    ventana_coef = tk.Toplevel()
    ventana_coef.title("Coeficientes de la Serie de Fourier")
    text = tk.Text(ventana_coef, width=60, height=30, font=("Courier New", 10))
    text.pack(padx=10, pady=10)
    text.insert(tk.END, f"a₀ = {a0:.6f}\n\n")
    for i, (a, b) in enumerate(zip(an, bn), 1):
        text.insert(tk.END, f"a{i} = {a:.6f},    b{i} = {b:.6f}\n")
    text.config(state="disabled")

def exportar_pdf(nombre_funcion, a0, an, bn, T, N, intervalo):
    c = canvas.Canvas("coeficientes_fourier.pdf", pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Aproximación por Serie de Fourier")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Función: {nombre_funcion}")
    c.drawString(50, height - 90, f"Intervalo: {intervalo}")
    c.drawString(50, height - 110, f"Periodo T: {T}")
    c.drawString(50, height - 130, f"Número de términos N: {N}")
    c.drawString(50, height - 160, f"a₀ = {a0:.6f}")
    y = height - 190
    for i, (a, b) in enumerate(zip(an, bn), 1):
        c.drawString(50, y, f"a{i} = {a:.6f},    b{i} = {b:.6f}")
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 12)
    c.save()
    messagebox.showinfo("PDF Generado", "Coeficientes guardados en coeficientes_fourier.pdf")

def animar_convergencia(funcion, nombre, T, max_N, muestras, x_min, x_max):
    x = np.linspace(x_min, x_max, muestras)
    y_real = np.vectorize(funcion)(x)
    fig, ax = plt.subplots(figsize=(10, 6))
    linea_aprox, = ax.plot([], [], lw=2, label='Aproximación Fourier')
    ax.plot(x, y_real, 'r--', label='Función Original')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-2, 2)
    ax.set_title(f"Convergencia de la Serie de Fourier para {nombre}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)

    def actualizar(n):
        a0, an, bn = calcular_coeficientes(funcion, T, n)
        y_aprox = [fourier_series(xi, a0, an, bn, n, T) for xi in x]
        linea_aprox.set_data(x, y_aprox)
        ax.set_title(f"N = {n}")
        return linea_aprox,

    anim = FuncAnimation(fig, actualizar, frames=range(1, max_N + 1), interval=300, blit=True)
    plt.show()

# === Interfaz ===
def graficar():
    try:
        idx = combo_funcion.current()
        nombre_funcion, funcion = funciones[idx]

        if entrada_intervalo.get().strip() != "[-3.14, 3.14]":
            messagebox.showinfo("Advertencia", "El intervalo fue ajustado automáticamente a [-3.14, 3.14]")
            entrada_intervalo.delete(0, tk.END)
            entrada_intervalo.insert(0, "[-3.14, 3.14]")
        x_min, x_max = -3.14, 3.14
        T = x_max - x_min

        N = int(entrada_N.get())
        if N > 1000:
            N = 1000
            entrada_N.delete(0, tk.END)
            entrada_N.insert(0, "1000")
            messagebox.showinfo("Advertencia", "El número de términos N no puede ser mayor a 1000. Se ajustó automáticamente.")

        muestras = int(entrada_muestras.get())
        if muestras > 1000:
            muestras = 1000
            entrada_muestras.delete(0, tk.END)
            entrada_muestras.insert(0, "1000")
            messagebox.showinfo("Advertencia", "El número de muestras no puede ser mayor a 1000. Se ajustó automáticamente.")

        a0, an, bn = calcular_coeficientes(funcion, T, N)
        graficar_fourier(funcion, nombre_funcion, a0, an, bn, N, T, muestras, x_min, x_max)
        mostrar_coeficientes(a0, an, bn)
        exportar_pdf(nombre_funcion, a0, an, bn, T, N, "[-3.14, 3.14]")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def mostrar_animacion():
    try:
        idx = combo_funcion.current()
        nombre_funcion, funcion = funciones[idx]

        if entrada_intervalo.get().strip() != "[-3.14, 3.14]":
            messagebox.showinfo("Advertencia", "El intervalo fue ajustado automáticamente a [-3.14, 3.14]")
            entrada_intervalo.delete(0, tk.END)
            entrada_intervalo.insert(0, "[-3.14, 3.14]")
        x_min, x_max = -3.14, 3.14
        T = x_max - x_min

        N = int(entrada_N.get())
        if N > 1000:
            N = 1000
            entrada_N.delete(0, tk.END)
            entrada_N.insert(0, "1000")
            messagebox.showinfo("Advertencia", "El número de términos N no puede ser mayor a 1000. Se ajustó automáticamente.")

        muestras = int(entrada_muestras.get())
        if muestras > 1000:
            muestras = 1000
            entrada_muestras.delete(0, tk.END)
            entrada_muestras.insert(0, "1000")
            messagebox.showinfo("Advertencia", "El número de muestras no puede ser mayor a 1000. Se ajustó automáticamente.")

        animar_convergencia(funcion, nombre_funcion, T, N, muestras, x_min, x_max)

    except Exception as e:
        messagebox.showerror("Error", str(e))

def analizar_audio():
    try:
        ruta = filedialog.askopenfilename(filetypes=[("Archivo WAV", "*.wav")])
        if not ruta:
            return
        sample_rate, data = wav.read(ruta)
        if data.ndim > 1:
            data = data[:, 0]
        N = len(data)
        T = 1.0 / sample_rate
        yf = np.fft.fft(data)
        xf = np.fft.fftfreq(N, T)[:N // 2]
        plt.figure(figsize=(10, 6))
        plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]), color='darkgreen')
        plt.title("Espectro de Fourier del archivo de audio")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Magnitud")
        plt.grid()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo analizar el audio:\n{e}")

# === Construcción de GUI ===
ventana = tk.Tk()
ventana.title("Serie de Fourier")
ventana.geometry("400x610")
ventana.resizable(False, False)

frame = tk.Frame(ventana, bg="#f4f4f9", bd=10, relief="solid", padx=15, pady=15)
frame.place(relx=0.5, rely=0.5, anchor="center")
font = ("Arial", 14)

widgets = [
    (tk.Label(frame, text="Selecciona una función:", font=font, bg="#f4f4f9"), 0),
    (ttk.Combobox(frame, values=[nombre for nombre, _ in funciones], font=font, width=22), 1),
    (tk.Label(frame, text="Intervalo (fijo en [-3.14, 3.14]):", font=font, bg="#f4f4f9"), 2),
    (tk.Entry(frame, font=font, width=22), 3),
    (tk.Label(frame, text="Número de términos N:", font=font, bg="#f4f4f9"), 4),
    (tk.Entry(frame, font=font, width=22), 5),
    (tk.Label(frame, text="Número de muestras:", font=font, bg="#f4f4f9"), 6),
    (tk.Entry(frame, font=font, width=22), 7),
    (tk.Button(frame, text="Graficar y Exportar PDF", font=("Arial", 16), command=graficar), 8),
    (tk.Button(frame, text="Visualizar Convergencia", font=("Arial", 14), command=mostrar_animacion), 9),
    (tk.Button(frame, text="Analizar Audio (.wav)", font=("Arial", 14), command=analizar_audio), 10)
]

for widget, row in widgets:
    widget.grid(row=row, column=0, pady=10, padx=10, sticky="ew")

combo_funcion = widgets[1][0]
combo_funcion.current(0)
entrada_intervalo = widgets[3][0]
entrada_intervalo.insert(0, "[-3.14, 3.14]")
entrada_N = widgets[5][0]
entrada_N.insert(0, "10")
entrada_muestras = widgets[7][0]
entrada_muestras.insert(0, "1000")

ventana.mainloop()
