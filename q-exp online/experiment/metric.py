class Metric:
    def __init__(self, name, key_values_dict, primary_key, experiment_manager):
        self.experiment_manager = experiment_manager
        self.keys = key_values_dict.keys()
        self.table_name = name
        self.experiment_manager.make_table(name, key_values_dict, primary_key)
        self.list_of_data = []

    def commit_to_database(self):
        if len(self.list_of_data) == 0:
            return
        self.experiment_manager.insert_values(self.table_name, self.keys, self.list_of_data)
        self.list_of_data = []

    def add_data(self, list_of_values):
        self.list_of_data.append(list_of_values)

    def clear_data(self):
        self.list_of_data = []
