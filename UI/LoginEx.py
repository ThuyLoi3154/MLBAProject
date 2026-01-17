from UI.Login import Ui_MainWindow
from UI.MainWindowEx import MainWindowEx
from PyQt6.QtWidgets import QMainWindow
import json
import os

class LoginEx(Ui_MainWindow):
    # def __init__(self, connector=None):
    #     self.connector = Connector()
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow = MainWindow
        self.pushButtonLogIn.clicked.connect(self.processLogin)
    def processLogin(self):
        # Read data from JSON file
        with open('D:\\project\\dataset\\user.json', 'r') as file:
            users = json.load(file)
        

        # Get username and password from line edits
        username = self.lineEdiUsername.text()
        password = self.lineEditPassword.text()

        for user in users:
            if username == user['username'] and password == user['pass']:
                self.MainWindow.close()
                self.Gui = MainWindowEx()
                self.Gui.setupUi(QMainWindow())
                self.Gui.show()
                return
            else:
                print('error')
                return
        
    def show(self):
        self.MainWindow.show()