fig = Figure(figsize=(3,3))
            a = fig.add_subplot(111)
            a.plot(amount, results_regular, color = "blue", label = "predicted")
            a.plot(amount, actual, color = "red", label = "actual")
            a.set_title("Forecasting Plot", fontsize = 10)
            canvas = FigureCanvasTkAgg(fig, master = self.master3)
            canvas.get_tk_widget().pack()
