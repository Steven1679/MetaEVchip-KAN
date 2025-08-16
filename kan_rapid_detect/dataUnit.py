class DataUnit(object):
    def __init__(self, args, col_index, excel_data=None):
        self.args = args
        self.col_index = col_index
        self.shift = None
        self.spectrum = None

        if excel_data is not None:
            self.get_data(excel_data)

    def get_data(self, excel_data):
        try:
            self.shift = [excel_data.iloc[0, self.col_index]]
            # save spectrum data
            self.spectrum = excel_data.iloc[1:, self.col_index].values
        except Exception as e:
            print(f"Error loading data for column {self.col_index}: {e}")