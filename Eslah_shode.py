import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from scipy.signal import hilbert


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = [fig.add_subplot(3, 4, i+1)
                     for i in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]]
        super(MatplotlibCanvas, self).__init__(fig)
        self.setParent(parent)
        self.plot_blank()

    def plot_blank(self):
        titles = ['Message Signal', 'Carrier Signal', 'AM Modulated Signal', 'DSB Modulated Signal', 'SSB Modulated Signal',
                  'LSSB/USSB Modulated Signal', 'SNR vs Gamma', 'VSB Modulated Signal', 'FM Modulated Signal', 'PM Modulated Signal']
        for ax, title in zip(self.axes, titles):
            ax.clear()
            ax.plot([], [])
            ax.set_title(title)
            ax.tick_params(axis='both', which='major', labelsize=8)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('PyQt5 with Matplotlib')
        self.setGeometry(100, 100, 800, 800)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)

        input_layout = QHBoxLayout()
        layout.addLayout(input_layout)

        input_details = [
            ("Sampling Freq", "250"),
            ("Carrier Freq", "10"),
            ("Carrier Amplitude", "1"),
            ("Message Freq", "5"),
            ("Message Amplitude", "1"),
            ("FM Kf", "10"),
            ("PM Kp", "1"),
            ("AM Mu", "0.5")
        ]

        self.inputs = {}
        for name, default_value in input_details:
            label = QLabel(f'{name}:')
            input_layout.addWidget(label)
            line_edit = QLineEdit(default_value)
            input_layout.addWidget(line_edit)
            self.inputs[name] = line_edit

        self.modulation_dropdown = QComboBox(self)
        self.modulation_dropdown.addItems(
            ["DSB", "VSB", "SSB", "LSSB", "USSB", "AM", "FM", "PM"])
        layout.addWidget(self.modulation_dropdown)

        update_button = QPushButton("Update Plot")
        update_button.clicked.connect(self.update_plot)
        layout.addWidget(update_button)

    def update_plot(self):
        try:
            Fs = float(self.inputs["Sampling Freq"].text())
            Fc = float(self.inputs["Carrier Freq"].text())
            Ac = float(self.inputs["Carrier Amplitude"].text())
            Fm = float(self.inputs["Message Freq"].text())
            Am = float(self.inputs["Message Amplitude"].text())
            Kf = float(self.inputs["FM Kf"].text())
            Kp = float(self.inputs["PM Kp"].text())
            Mu = float(self.inputs["AM Mu"].text())
        except ValueError:
            self.show_error_message(
                "Invalid input! Please enter numeric values.")
            return

        T = np.arange(0, 1, 1/Fs)
        MessageSig = Am * np.sin(2 * np.pi * Fm * T)
        CarrSig = Ac * np.sin(2 * np.pi * Fc * T)

        AMModSig = Ac * (1 + Mu * MessageSig) * np.cos(2 * np.pi * Fc * T)
        FMModSig = np.sin(2 * np.pi * Fc * T + 2 * np.pi *
                          Kf * np.cumsum(MessageSig) / Fs)
        PMModSig = np.sin(2 * np.pi * Fc * T + Kp * MessageSig)
        DSBModSig = MessageSig * CarrSig

        HilbertCarrSig = np.imag(hilbert(CarrSig))
        HilbertMsgSig = np.imag(hilbert(MessageSig))
        VSBModSig = MessageSig * CarrSig + HilbertMsgSig * HilbertCarrSig
        SSBModSig = MessageSig * \
            np.cos(2 * np.pi * Fc * T) - HilbertMsgSig * \
            np.sin(2 * np.pi * Fc * T)
        LSSBModSig = SSBModSig * np.heaviside(2 * np.pi * Fc * T - np.pi/2, 0)
        USSBModSig = SSBModSig * np.heaviside(np.pi/2 - 2 * np.pi * Fc * T, 0)

        selected_modulation = self.modulation_dropdown.currentText()
        ModSig = {
            "DSB": DSBModSig,
            "VSB": VSBModSig,
            "SSB": SSBModSig,
            "LSSB": LSSBModSig,
            "USSB": USSBModSig,
            "AM": AMModSig,
            "FM": FMModSig,
            "PM": PMModSig
        }[selected_modulation]

        GammaVals = np.arange(0, 10.1, 0.1)
        SNRVals = np.zeros(len(GammaVals))

        for i, gamma in enumerate(GammaVals):
            Noise = gamma * np.random.randn(len(ModSig))
            NoisySig = ModSig + Noise
            SigPow = np.mean(ModSig ** 2)
            NoisePow = np.mean(Noise ** 2)
            SNRVals[i] = 10 * np.log10(SigPow / (NoisePow + 1e-7))

        self.plot_signals(T, MessageSig, CarrSig, AMModSig, FMModSig, PMModSig,
                          DSBModSig, VSBModSig, SSBModSig, LSSBModSig, USSBModSig, GammaVals, SNRVals, selected_modulation)

    def plot_signals(self, T, MessageSig, CarrSig, AMModSig, FMModSig, PMModSig,
                     DSBModSig, VSBModSig, SSBModSig, LSSBModSig, USSBModSig, GammaVals, SNRVals, selected_modulation):
        colors = ['b', 'g', 'r', 'c', 'm', 'y',
                  'k', 'orange', 'purple', 'brown']

        self.canvas.axes[0].clear()
        self.canvas.axes[0].plot(T, MessageSig, color=colors[0])
        self.canvas.axes[0].set_title('Message Signal')
        self.canvas.axes[0].tick_params(
            axis='both', which='major', labelsize=8)

        self.canvas.axes[1].clear()
        self.canvas.axes[1].plot(T, CarrSig, color=colors[1])
        self.canvas.axes[1].set_title('Carrier Signal')
        self.canvas.axes[1].tick_params(
            axis='both', which='major', labelsize=8)

        self.canvas.axes[2].clear()
        self.canvas.axes[2].plot(T, AMModSig, color=colors[2])
        self.canvas.axes[2].set_title('AM Modulated Signal')
        self.canvas.axes[2].tick_params(
            axis='both', which='major', labelsize=8)

        self.canvas.axes[3].clear()
        self.canvas.axes[3].plot(T, DSBModSig, color=colors[3])
        self.canvas.axes[3].set_title('DSB Modulated Signal')
        self.canvas.axes[3].tick_params(
            axis='both', which='major', labelsize=8)

        self.canvas.axes[4].clear()
        self.canvas.axes[4].plot(T, SSBModSig, color=colors[4])
        self.canvas.axes[4].set_title('SSB Modulated Signal')
        self.canvas.axes[4].tick_params(
            axis='both', which='major', labelsize=8)

        self.canvas.axes[5].clear()
        self.canvas.axes[5].plot(T, LSSBModSig, color=colors[5], label='LSSB')
        self.canvas.axes[5].plot(T, USSBModSig, color=colors[6], label='USSB')
        self.canvas.axes[5].set_title('LSSB/USSB Modulated Signal')
        self.canvas.axes[5].legend()
        self.canvas.axes[5].tick_params(
            axis='both', which='major', labelsize=8)

        self.canvas.axes[6].clear()
        self.canvas.axes[6].plot(GammaVals, SNRVals, color=colors[7])
        self.canvas.axes[6].set_title(f'SNR vs Gamma ({selected_modulation})')
        self.canvas.axes[6].tick_params(
            axis='both', which='major', labelsize=8)

        self.canvas.axes[7].clear()
        self.canvas.axes[7].plot(T, VSBModSig, color=colors[8])
        self.canvas.axes[7].set_title('VSB Modulated Signal')
        self.canvas.axes[7].tick_params(
            axis='both', which='major', labelsize=8)

        self.canvas.axes[8].clear()
        self.canvas.axes[8].plot(T, FMModSig, color=colors[9])
        self.canvas.axes[8].set_title('FM Modulated Signal')
        self.canvas.axes[8].tick_params(
            axis='both', which='major', labelsize=8)

        self.canvas.axes[9].clear()
        self.canvas.axes[9].plot(T, PMModSig, color=colors[2])
        self.canvas.axes[9].set_title('PM Modulated Signal')
        self.canvas.axes[9].tick_params(
            axis='both', which='major', labelsize=8)

        self.canvas.draw()

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
