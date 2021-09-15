import xlrd


xls_file = xlrd.open_workbook("gnhbsjnew.xls")

xls_sheet = xls_file.sheets()[0]
row_value = xls_sheet.row_values(1)
print row_value