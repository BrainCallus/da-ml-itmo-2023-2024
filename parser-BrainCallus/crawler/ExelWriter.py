import xlsxwriter
from xlsxwriter.worksheet import Worksheet


class ExelWriter:
    default_separator = ';'
    default_sheet_name = 'list 0'
    test_directory = 'tests/'

    def __init__(self, name):
        self.workbook = xlsxwriter.Workbook(self.test_directory + name)
        self.has_title_row = False

    def set_title_row(self, row, sheet_name=default_sheet_name):
        sheet = self.workbook.get_worksheet_by_name(sheet_name)
        if sheet is None:
            sheet = self.workbook.add_worksheet(sheet_name)
        self.write_row(row, sheet, 0, 0)
        self.has_title_row = True

    def write(self, collection, sheet_name=default_sheet_name, separator=default_separator, start_column=0):

        sheet = self.workbook.get_worksheet_by_name(sheet_name)
        row = 1 if self.has_title_row else 0
        for item in collection:
            if hasattr(item, "__iter__") and not isinstance(item, str):
                self.write_row(item, sheet, row, start_column)

            elif isinstance(item, str):
                self.write_row(item.split(separator), sheet, row, start_column)

            else:
                sheet.write(row, 0, item)

            row = row + 1

    def write_single(self, row, col, content, sheet_name=default_sheet_name):
        sheet = self.workbook.get_worksheet_by_name(sheet_name)
        sheet.write(row, col, content)

    @staticmethod
    def write_row(row_elements, sheet, row, start_column):
        if isinstance(sheet, Worksheet):
            col = start_column
            for element in row_elements:
                sheet.write(row, col, element)
                col = col + 1

    def close(self):
        self.workbook.close()
