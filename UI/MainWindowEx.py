import random
from random import random
import plotly.graph_objects as go
import pandas as pd

from PyQt6 import QtGui, QtCore
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QPixmap
from PyQt6.QtWidgets import QMessageBox, QProgressBar, QTableWidgetItem, QMainWindow, QDialog, QComboBox, QPushButton, QCheckBox, \
    QListWidgetItem
from matplotlib import pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.linear_model import LinearRegression

from UI.MainWindow import Ui_MainWindow
import traceback


import matplotlib

from matplotlib.figure import Figure


from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from Models.RFMcalculation import heatmap
from Models.kmean import df_cluster_segment1, RFM_df4, fig, figD
from Models.cluster_analysis import figR, figH
from Models.recommendation import customer_data_with_recommendations, filter_customer_data
from Models.chartoption import plot_product_category, plot_product_specific, plot_all_categories
from Models.eda import figCS, figO, figOS, figPM, figPC, figS, figORS, figFC, figRE, figGEO


class MainWindowEx(Ui_MainWindow):
    def __init__(self):
        pass
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow=MainWindow

        # TREE MAP
        # Create a FigureCanvas from the heatmap figure
        canvas = FigureCanvas(heatmap)
        toolbarC = NavigationToolbar(canvas,self.MainWindow)
        self.verticalLayoutTreeMap.addWidget(toolbarC)
        self.verticalLayoutTreeMap.addWidget(canvas)

        # # TABLE
        # Fill the table with data from df_cluster_segment
        self.tableWidgetStatistic.setRowCount(df_cluster_segment1.shape[0])
        self.tableWidgetStatistic.setColumnCount(df_cluster_segment1.shape[1])

        for i in range(df_cluster_segment1.shape[0]):
            for j in range(df_cluster_segment1.shape[1]):
                self.tableWidgetStatistic.setItem(i, j, QTableWidgetItem(str(df_cluster_segment1.iat[i, j])))
        # Set the column headers
        self.tableWidgetStatistic.setHorizontalHeaderLabels(df_cluster_segment1.columns)
        # Set the row headers
        self.tableWidgetStatistic.setVerticalHeaderLabels(df_cluster_segment1.index.astype(str))

        # 3D PLOT
        plot3d = FigureCanvas(fig)
        toolbar = NavigationToolbar(plot3d,self.MainWindow)
        self.verticalLayout3DModel.addWidget(toolbar)
        self.verticalLayout3DModel.addWidget(plot3d)

        # DISTRIBUTION
        plotD = FigureCanvas(figD)
        toolbarD = NavigationToolbar(plotD, self.MainWindow)
        self.verticalLayoutDistribution_2.addWidget(toolbarD)
        self.verticalLayoutDistribution_2.addWidget(plotD)

        # RADAR CHART
        plotR = FigureCanvas(figR)
        toolbarR = NavigationToolbar(plotR, self.MainWindow)
        self.verticalLayoutRadiusAnalysis_7.addWidget(toolbarR)
        self.verticalLayoutRadiusAnalysis_7.addWidget(plotR)

        # HIST CHART
        plotH = FigureCanvas(figH)
        toolbarH = NavigationToolbar(plotH, self.MainWindow)
        self.verticalLayoutRadiusAnalysis.addWidget(toolbarH)
        self.verticalLayoutRadiusAnalysis.addWidget(plotH)

        # RECOMMENDATION
        self.pushButtonGo.clicked.connect(self.recommendresults)

        # LINEAR REGRESSION
        self.pushButtonConfirm.clicked.connect(self.linearregression)

        # CHART OPTIONS
        self.pushButtonSubmit.clicked.connect(self.plot_category)

        # REPORT TAB
        self.pushButtonStatisticsOnCustomersandSellers.clicked.connect(self.StatCustomernSeller)
        self.pushButtonStatisticsOnOrders.clicked.connect(self.StatOrders)
        self.pushButtonStatisticsOnOrderValuesAndShippingCost.clicked.connect(self.StatOrdernShipping)
        self.pushButtonPurchaseValueByCategory.clicked.connect(self.StatPurchaseCate)
        self.pushButtonStatisticsOnSelling.clicked.connect(self.StatSelling)
        self.pushButtonStatisticOnOrderStatus.clicked.connect(self.StatOrderStatus)
        self.pushButtonStatisticsOnFreightTimeAndCost.clicked.connect(self.StatFreightnCost)
        self.pushButtonStatisticsOnReviews.clicked.connect(self.StatReviews)
        self.pushButtonPaymentMethods.clicked.connect(self.StatPaymentMethod)
        self.pushButtonStatisticsOnSalesGeolocation.clicked.connect(self.StatGeolocation)

    def show(self):
        self.MainWindow.show()
    def recommendresults(self):
        # Get the customer ID from the input field
        customer_id = self.lineEditCustomer_unique_id.text()
        # Filter the customer data based on the customer ID
        filtered_data = filter_customer_data(customer_data_with_recommendations, customer_id)
        # Display the recommendations in the list widget
        self.listWidgetRecommendedProducts.clear()
        # Add the filtered data to the QListWidget
        for index, row in filtered_data.iterrows():
            item = QListWidgetItem(str(row))
            self.listWidgetRecommendedProducts.addItem(item)
    def linearregression(self):
        regr = load('D:\\project\\linear_regression_model.joblib')
        # Get the input values from the input fields
        price = self.lineEditPrice.text()
        freight_value = self.lineEditFreight_value.text()
        product_name_lenght = self.lineEditProduct_name_lenght.text()
        product_photos_qty = self.lineEditProduct_photos_qty.text()
        estimated_delivery_time = self.lineEditEstimated_delivery_time.text()

        # Create a dictionary with the input values
        input_data = {
            'price': [float(price)],
            'freight_value': [float(freight_value)],
            'product_name_lenght': [float(product_name_lenght)],
            'product_photos_qty': [float(product_photos_qty)],
            'estimated_delivery_time': [float(estimated_delivery_time)]
        }

        # Create a DataFrame from the input data
        input_df = pd.DataFrame(input_data)

        # Make a prediction using the linear regression model
        prediction = regr.predict(input_df)

        # Display the prediction in the lineEditPayment_value
        self.lineEditPayment_value.setText(str(prediction[0]))
    def plot_category(self):
        # Get the selected category and time period
       
        category_name = self.lineEditProduct_category_name.text()
        product_id = self.lineEditSpecificProduct.text()

        while self.verticalLayoutChart.count():
            widget = self.verticalLayoutChart.takeAt(0).widget()
            if widget is not None: 
                widget.deleteLater()

        if self.radioButtonProduct_category_name.isChecked() and self.radioButtonWeek.isChecked():
            set_week = 'week'
            plot1 = plot_product_category(category_name, set_week)
            chart1 = FigureCanvas(plot1)
            toolbar1 = NavigationToolbar(chart1, self.MainWindow)
            self.verticalLayoutChart.addWidget(toolbar1)
            self.verticalLayoutChart.addWidget(chart1)

        if self.radioButtonSpecificProduct.isChecked() and self.radioButtonWeek.isChecked():
            set_week = 'week'
            plot2 = plot_product_specific(product_id, set_week)
            chart2 = FigureCanvas(plot2)
            toolbar2 = NavigationToolbar(chart2, self.MainWindow)
            self.verticalLayoutChart.addWidget(toolbar2)
            self.verticalLayoutChart.addWidget(chart2)

        if self.radioButtonAllProducts.isChecked() and self.radioButtonWeek.isChecked():
            set_week = 'week'
            plot3 = plot_all_categories(set_week)
            chart3 = FigureCanvas(plot3)
            toolbar3 = NavigationToolbar(chart3, self.MainWindow)
            self.verticalLayoutChart.addWidget(toolbar3)
            self.verticalLayoutChart.addWidget(chart3)

        if self.radioButtonProduct_category_name.isChecked() and self.radioButtonMonth.isChecked():
            set_month = 'month'
            plot4 = plot_product_category(category_name, set_month)
            chart4 = FigureCanvas(plot4)
            toolbar4 = NavigationToolbar(chart4, self.MainWindow)
            self.verticalLayoutChart.addWidget(toolbar4)
            self.verticalLayoutChart.addWidget(chart4)

        if self.radioButtonSpecificProduct.isChecked() and self.radioButtonMonth.isChecked():
            set_month = 'month'
            plot5 = plot_product_specific(product_id, set_month)
            chart5 = FigureCanvas(plot5)
            toolbar5 = NavigationToolbar(chart5, self.MainWindow)
            self.verticalLayoutChart.addWidget(toolbar5)
            self.verticalLayoutChart.addWidget(chart5)

        if self.radioButtonAllProducts.isChecked() and self.radioButtonMonth.isChecked():
            set_month = 'month'
            plot6 = plot_all_categories(set_month)
            chart6 = FigureCanvas(plot6)
            toolbar6 = NavigationToolbar(chart6, self.MainWindow)
            self.verticalLayoutChart.addWidget(toolbar6)
            self.verticalLayoutChart.addWidget(chart6)

    def StatCustomernSeller(self):
        while self.verticalLayout.count():
            item = self.verticalLayout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None: 
                    widget.deleteLater()
        plotCS = FigureCanvas(figCS)
        toolbarCS = NavigationToolbar(plotCS, self.MainWindow)
        self.verticalLayout.addWidget(toolbarCS)
        self.verticalLayout.addWidget(plotCS)

    def StatOrders(self):
        while self.verticalLayout.count():
            item = self.verticalLayout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None: 
                    widget.deleteLater()
        plotO = FigureCanvas(figO)
        toolbarO = NavigationToolbar(plotO, self.MainWindow)
        self.verticalLayout.addWidget(toolbarO)
        self.verticalLayout.addWidget(plotO)

    def StatOrdernShipping(self):
        while self.verticalLayout.count():
            item = self.verticalLayout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None: 
                    widget.deleteLater()
        plotOS = FigureCanvas(figOS)
        toolbarOS = NavigationToolbar(plotOS, self.MainWindow)
        self.verticalLayout.addWidget(toolbarOS)
        self.verticalLayout.addWidget(plotOS)
    
    def StatPurchaseCate(self):
        while self.verticalLayout.count():
            item = self.verticalLayout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None: 
                    widget.deleteLater()
        plotPC = FigureCanvas(figPC)
        toolbarPC = NavigationToolbar(plotPC, self.MainWindow)
        self.verticalLayout.addWidget(toolbarPC)
        self.verticalLayout.addWidget(plotPC)

    def StatSelling(self):
        while self.verticalLayout.count():
            item = self.verticalLayout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None: 
                    widget.deleteLater()
        plotS = FigureCanvas(figS)
        toolbarS = NavigationToolbar(plotS, self.MainWindow)
        self.verticalLayout.addWidget(toolbarS)
        self.verticalLayout.addWidget(plotS)

    def StatOrderStatus(self):
        while self.verticalLayout.count():
            item = self.verticalLayout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None: 
                    widget.deleteLater()
        plotORS = FigureCanvas(figORS)
        toolbarORS = NavigationToolbar(plotORS, self.MainWindow)
        self.verticalLayout.addWidget(toolbarORS)
        self.verticalLayout.addWidget(plotORS)
        
    def StatFreightnCost(self):
        while self.verticalLayout.count():
            item = self.verticalLayout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None: 
                    widget.deleteLater()
        plotFC = FigureCanvas(figFC)
        toolbarFC = NavigationToolbar(plotFC, self.MainWindow)
        self.verticalLayout.addWidget(toolbarFC)
        self.verticalLayout.addWidget(plotFC)

    def StatReviews(self):
        while self.verticalLayout.count():
            item = self.verticalLayout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None: 
                    widget.deleteLater()
        plotRE = FigureCanvas(figRE)
        toolbarRE = NavigationToolbar(plotRE, self.MainWindow)
        self.verticalLayout.addWidget(toolbarRE)
        self.verticalLayout.addWidget(plotRE)   

    def StatPaymentMethod(self):
        while self.verticalLayout.count():
            item = self.verticalLayout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None: 
                    widget.deleteLater()
        plotPM = FigureCanvas(figPM)
        toolbarPM = NavigationToolbar(plotPM, self.MainWindow)
        self.verticalLayout.addWidget(toolbarPM)
        self.verticalLayout.addWidget(plotPM)   

    def StatGeolocation(self):
        while self.verticalLayout.count():
            item = self.verticalLayout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None: 
                    widget.deleteLater()
        plotGEO = FigureCanvas(figGEO)
        toolbarGEO = NavigationToolbar(plotGEO, self.MainWindow)
        self.verticalLayout.addWidget(toolbarGEO)
        self.verticalLayout.addWidget(plotGEO)  