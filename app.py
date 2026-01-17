from PyQt6.QtWidgets import QApplication, QMainWindow

from UI.LoginEx import LoginEx

qApp=QApplication([])
qmainWindow=QMainWindow()
window=LoginEx()
window.setupUi(qmainWindow)
window.show()
qApp.exec()
