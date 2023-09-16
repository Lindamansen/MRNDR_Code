#encoding=utf-8
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication,QWidget, QPushButton, QLabel, QInputDialog, QTextBrowser,QTableWidget,QTableView,QTabWidget,QTableWidgetItem,QComboBox,QFileDialog,QDirModel)
from PyQt5.QtGui import *
import pandas as pd

data_tui=pd.read_csv("data_tui.csv")
sample_text_tui=pd.read_csv("sample_text_tui_1.csv")
text_tui=pd.read_csv("text_tui.csv")

drug_name=[]
disease_name=[]
score_all=[]
mode_all=[]
for i in data_tui.iloc[:,1]:
    drug__name=i.split("药")[0]
    disease__name=i.split("病")[1]
    # score=i.split("相关性分数:")[1]
    mode=i.split("物")[1].split("治")[0]
    drug_name.append(drug__name)
    disease_name.append(disease__name)
    # score_all.append(score)
    mode_all.append(mode)
for j in data_tui.iloc[:,2]:
    score_all.append(round(j, 6))
text_number=[]
for i in text_tui.iloc[:,0]:
    text_number.append(i)
# link_number=[]
# for i in link_dataset.iloc[:,1]:
#     link_number.append(i)
Drug_Disease=[]
for i in range(len(text_number)):
    drugs_diseases=drug_name[i]+"||"+disease_name[i]+"||"+str(text_number[i])
    Drug_Disease.append(drugs_diseases)
def sigmoid(x):
    return 1/(1+ np.exp(-x))
sum_axis=[]
location=[]

sample_text=pd.DataFrame(sample_text_tui.iloc[:,0:2].values)
for i in range(len(sample_text)):
    Rare_match=text_tui.iloc[i][0]/sample_text.iloc[i][0]+text_tui.iloc[i][0]/sample_text.iloc[i][1]
    # Rare_match_inv =  (sample_text.iloc[i][0] +sample_text.iloc[i][1])/text_tui.iloc[i][0]
    sum_axis.append(Rare_match)
    location.append(i)

# for i in range(len(text_number)):
#     sum=text_number[i]+score_all[i]
#     axis=sigmoid(sum)
#     sum_axis.append(axis)
#     location.append(i)
sum_location=zip(sum_axis,location)
sum_locat_all=sorted(sum_location,key=lambda x: x[0])
result = zip(*sum_locat_all)
Rare_values, location= [list(x) for x in result]
rare_values=['UnKnow' if i ==0 else i for i in Rare_values]

Drug_name=[]
Disease_name=[]
Text_number=[]
Score_all=[]
Mode_all=[]

for i in location:
    Drug_name.append(drug_name[i])
    Disease_name.append(disease_name[i])
    Text_number.append(text_number[i])
    Mode_all.append(mode_all[i])
    Score_all.append(score_all[i])



new_drug_name=list(dict.fromkeys(drug_name))
new_disease_name=list(dict.fromkeys(disease_name))
new_name=np.append(new_drug_name,new_disease_name)



class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df.copy()

    def toDataFrame(self):
        return self._df.copy()

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()
        return QtCore.QVariant(str(self._df.iloc[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()


class myWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(myWindow, self).__init__(parent)
        self.centralwidget  = QtWidgets.QWidget(self)
        self.lineEdit       = QtWidgets.QLineEdit(self.centralwidget)
        self.view           = QtWidgets.QTableView(self.centralwidget)
        self.comboBox       = QtWidgets.QComboBox(self.centralwidget)
        self.label          = QtWidgets.QLabel(self.centralwidget)

        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.view, 1, 0, 1, 3)
        self.gridLayout.addWidget(self.comboBox, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.setCentralWidget(self.centralwidget)
        self.label.setText("Regex Filter")

        self.load_sites()
        self.comboBox.addItems(["{0}".format(col) for col in self.model._df.columns])

        self.lineEdit.textChanged.connect(self.on_lineEdit_textChanged)
        self.comboBox.currentIndexChanged.connect(self.on_comboBox_currentIndexChanged)

        self.horizontalHeader = self.view.horizontalHeader()
        self.horizontalHeader.sectionClicked.connect(self.on_view_horizontalHeader_sectionClicked)
        self.initUI()

    def load_sites(self):
        df = pd.DataFrame({'药物':Drug_name,
                           '疾病':Disease_name,
                           '相关性分数':Score_all,
                           '是否已知':Mode_all,
                           '相关文献数量':Text_number,
                           '罕见值':rare_values})
        self.model = PandasModel(df)
        self.proxy = QtCore.QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.view.setModel(self.proxy)
        self.view.resizeColumnsToContents()

    @QtCore.pyqtSlot(int)
    def on_view_horizontalHeader_sectionClicked(self, logicalIndex):

        self.logicalIndex   = logicalIndex
        self.menuValues     = QtWidgets.QMenu(self)
        self.signalMapper   = QtCore.QSignalMapper(self)
        self.comboBox.blockSignals(True)
        self.comboBox.setCurrentIndex(self.logicalIndex)
        self.comboBox.blockSignals(True)

        valuesUnique = self.model._df.iloc[:, self.logicalIndex].unique()
        actionAll = QtWidgets.QAction("All", self)
        actionAll.triggered.connect(self.on_actionAll_triggered)
        self.menuValues.addAction(actionAll)
        self.menuValues.addSeparator()
        for actionNumber, actionName in enumerate(sorted(list(set(valuesUnique)))):
            action = QtWidgets.QAction(actionName, self)
            self.signalMapper.setMapping(action, actionNumber)
            action.triggered.connect(self.signalMapper.map)
            self.menuValues.addAction(action)
        self.signalMapper.mapped.connect(self.on_signalMapper_mapped)
        headerPos = self.view.mapToGlobal(self.horizontalHeader.pos())
        posY = headerPos.y() + self.horizontalHeader.height()
        posX = headerPos.x() + self.horizontalHeader.sectionPosition(self.logicalIndex)
        self.menuValues.exec_(QtCore.QPoint(posX, posY))

    @QtCore.pyqtSlot()
    def on_actionAll_triggered(self):
        filterColumn = self.logicalIndex
        filterString = QtCore.QRegExp(  "",
                                        QtCore.Qt.CaseInsensitive,
                                        QtCore.QRegExp.RegExp
                                        )

        self.proxy.setFilterRegExp(filterString)
        self.proxy.setFilterKeyColumn(filterColumn)

    @QtCore.pyqtSlot(int)
    def on_signalMapper_mapped(self, i):
        stringAction = self.signalMapper.mapping(i).text()
        filterColumn = self.logicalIndex
        filterString = QtCore.QRegExp(  stringAction,
                                        QtCore.Qt.CaseSensitive,
                                        QtCore.QRegExp.FixedString
                                        )

        self.proxy.setFilterRegExp(filterString)
        self.proxy.setFilterKeyColumn(filterColumn)

    @QtCore.pyqtSlot(str)
    def on_lineEdit_textChanged(self, text):
        search = QtCore.QRegExp(    text,
                                    QtCore.Qt.CaseInsensitive,
                                    QtCore.QRegExp.RegExp
                                    )

        self.proxy.setFilterRegExp(search)

    @QtCore.pyqtSlot(int)
    def on_comboBox_currentIndexChanged(self, index):
        self.proxy.setFilterKeyColumn(index)
    def initUI(self):
        self.lb1 = QLabel('名称：', self)
        self.lb1.move(1450, 50)
        self.lb2 = QLabel('药物_疾病_名称等', self)
        self.lb2.resize(350,30)
        self.lb2.move(1520, 50)
        self.bt1 = QPushButton('修改名称', self)
        self.bt1.move(1750, 50)
        self.bt1.clicked.connect(self.showDialog)
        self.lab1 =QLabel(self)
        self.h=QLabel(self)
        self.h.setText("知识图谱")
        self.h.move(1250,265)
        self.lab1.setText("显示图片")
        self.lab1.setFixedSize(600, 500)
        self.lab1.move(1250, 300)
        self.lab1.setStyleSheet("QLabel{background:grey;}")
        self.lab2=QLabel(self)
        self.lab3=QPushButton("Advanced",self)
        self.lab3.move(1750,150)
        self.lab3.clicked.connect(self.showText)
        self.lb3=QTextBrowser(self)
        self.lb3.move(1450, 120)
        self.lb3.resize(250,30)
        self.lb4 = QTextBrowser(self)
        self.lb4.move(1450, 170)
        self.lb4.resize(250,30)
        self.lab4=QLabel(self)
        self.lab4.setText("+")
        self.lab4.move(1580,145)
        self.lab5 = QLabel(self)
        self.lb = QPushButton(self)
        self.lb.setText("文献地址")
        self.lb.move(1250, 50)
        self.lb.clicked.connect(self.showLink)
        self.l = QLabel(self)
        self.save=QPushButton(self)
        self.save.setText("Save")
        self.save.move(1350,265)
        self.save.resize(90,30)
        self.save.clicked.connect(self.showimage)
        self.button_in = QPushButton('放大', self)
        self.button_in.move(1500, 265)
        self.button_in.resize(90,30)
        self.button_out = QPushButton('放小', self)
        self.button_out.clicked.connect(self.on_zoom_out)
        self.button_out.move(1650, 265)
        self.button_out.resize(90, 30)
        self.button_in.clicked.connect(self.on_zoom_in)
        self.button_out.clicked.connect(self.on_zoom_out)
    def showDialog(self):
        sender = self.sender()
        if sender == self.bt1:
            text, ok = QInputDialog.getItem(self, '修改名称', '请输入名称：',new_name)
            if ok:
                self.lb2.setText(text)
                self.showImage = QPixmap("./data/Kg_pyqt5/{}.png".format(text))
                self.height = self.showImage.height()
                self.lab1.setPixmap(self.showImage.scaled(600,500))
                self.lab2.setText("{}相关文献网址""<a href='https://pubmed.ncbi.nlm.nih.gov/?term={}'>""点击此处</a".format(text,text.replace(" ", "+")))
                self.lab2.resize(600, 20)
                self.lab2.move(1450, 85)
                self.lab2.setOpenExternalLinks(True)
    def showText(self):
        sender = self.sender()
        if sender == self.lab3:
            text_1, ok = QInputDialog.getItem(self, '修改名称', '请输入名称：', new_name)
            text_2, ok = QInputDialog.getItem(self, '修改名称', '请输入名称：', new_name)
            if ok:
                self.lb3.setText(text_1)
                self.lb4.setText(text_2)
                self.lab5.setText("{}..+{}..相关文献网址""<a href='https://pubmed.ncbi.nlm.nih.gov/?term={}+AND+{}&sort='>""点击此处</a".format(text_1,text_2,text_1.replace(" ","+"),text_2.replace(" ","+")))
                self.lab5.resize(600, 20)
                self.lab5.move(1450, 230)
                self.lab5.setOpenExternalLinks(True)
    def showLink(self):
        sender = self.sender()
        if sender == self.lb:
            link, ok = QInputDialog.getItem(self, '修改链接', '请输入链接：', Drug_Disease)
            if ok:
                self.l.setText(link)
                self.l.setText("<a href='https://pubmed.ncbi.nlm.nih.gov/?term={}+AND+{}&sort='>""点击链接</a".format(link.split("||")[0].replace(" ", "+"), link.split("||")[1].replace(" ", "+")))
                self.l.resize(200, 20)
                self.l.move(1250, 100)
                self.l.setOpenExternalLinks(True)
    def showimage(self):
        sender = self.sender()
        if sender == self.save:
            dir_path,ok=QFileDialog.getSaveFileName(self,filter='(*.png)')
            self.showImage.save("{}".format(dir_path))
    def on_zoom_in(self):
        self.height += 100
        self.resize_image()
    def on_zoom_out(self):
        self.height -= 100
        self.resize_image()
    def resize_image(self):
        scaled_pixmap = self.showImage.scaledToHeight(self.height)
        self.lab1.setPixmap(scaled_pixmap)



if __name__ == "__main__":
    import sys
    app  = QtWidgets.QApplication(sys.argv)
    main = myWindow()
    main.show()
    main.resize(2100,850)
    sys.exit(app.exec_())