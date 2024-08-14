import sys
import json
import random
import time

import mysql.connector
from omegaconf import DictConfig, ListConfig


class ExperimentManager:
    '''
    Class to manage and log experiments to a mysql server
    '''

    def __init__(self, experiment_name, parameters_dict, compute_canada_username, db_data):

        self.db_data = db_data

        self.db_name = compute_canada_username + "_" + experiment_name
        while (True):
            try:
                conn = mysql.connector.connect(
                    host=self.db_data["ip"],
                    user=self.db_data["username"],
                    password=self.db_data["password"]
                )
                break
            except:
                print("Database server not responding; busy waiting")
                time.sleep((random.random() + 0.2) * 5)

        sql_run = conn.cursor()
        try:
            sql_run.execute("CREATE DATABASE " + self.db_name + ";")
        except:
            pass
        sql_run.execute("USE " + self.db_name + ";")

        conn.close()

        self.command_args = "python " + " ".join(sys.argv)
        self.name_initial = experiment_name

        self.run = parameters_dict["run"]
        ret = self.make_table("runs", parameters_dict, ["run"])
        self.insert_value("runs", parameters_dict)

    def get_connection(self):
        while (True):
            try:
                conn = mysql.connector.connect(
                    host=self.db_data["ip"],
                    user=self.db_data["username"],
                    password=self.db_data["password"]
                )
                break
            except:
                print("Database server not responding; busy waiting")
                time.sleep((random.random() + 0.2) * 5)

        sql_run = conn.cursor()
        sql_run.execute("USE " + self.db_name + ";")
        return conn, sql_run

    def make_table(self, table_name, data_dict, primary_key):

        conn, sql_run = self.get_connection()

        table = "CREATE TABLE " + table_name + " ("
        counter = 0
        for a in data_dict:
            if type(data_dict[a]) is int or type(data_dict[a]) is float:
                table = table + a + " real"
            elif isinstance(data_dict[a], (DictConfig, ListConfig, dict, list)):
                table = table + a + " json"
            else:
                table = table + a + " text"

            counter += 1
            if counter != len(data_dict):
                table += ", "
        if primary_key is not None:
            table += " ".join([",", "PRIMARY KEY(", ",".join(primary_key)]) + ")"
        table = table + ");"
        try:
            sql_run.execute(table)
            conn.commit()
            conn.close()
            return True
        except:
            print("Failed creating table ", table_name, ", perhaps it already exists?")
            conn.close()
            return False

    def insert_value(self, table_name, data_dict):

        conn, sql_run = self.get_connection()
        values = []
        for val in tuple(data_dict.values()):
            v = str(val)
            if isinstance(val, (DictConfig, ListConfig, dict, list)):
                v = v.replace("'", "")
                v = v.replace("{", "\"{")
                v = v.replace("[", "\"[")
                v = v.replace("}", "}\"")
                v = v.replace("]", "]\"")
            values.append(v)
        query = " ".join(["INSERT INTO", table_name, str(tuple(data_dict.keys())).replace("'", ""), "VALUES",
                          str(tuple(values))]) + ";"
        sql_run.execute(query)
        conn.commit()
        conn.close()

    def insert_values(self, table_name, keys, value_list):
        conn, sql_run = self.get_connection()
        strin = "("
        counter = 0
        non_number_idxes = []
        for idx, val in enumerate(value_list[0]):
            counter += 1
            strin += "%s"
            if counter != len(value_list[0]):
                strin += ","
        strin += ");"

        all_values = []
        for row in value_list:
            values = []
            for val in row:
                v = str(val)
                if isinstance(val, (DictConfig, ListConfig, dict, list)):
                    v = v.replace("'", "")
                    v = v.replace("{", "\"{")
                    v = v.replace("[", "\"[")
                    v = v.replace("}", "}\"")
                    v = v.replace("]", "]\"")
                values.append(v)
            all_values.append(tuple(values))

        query = " ".join(
            ["INSERT INTO", table_name, str(tuple(keys)).replace("'", ""), "VALUES", strin])

        sql_run.executemany(query, all_values)
        conn.commit()
        conn.close()

    def insert_values_old(self, table_name, keys, value_list):
        conn, sql_run = self.get_connection()
        strin = "("
        counter = 0
        non_number_idxes = []
        for idx, val in enumerate(value_list[0]):
            counter += 1
            strin += "%s"
            if counter != len(value_list[0]):
                strin += ","
            # find non-number idxes
            if not type(val) in [int, float]:
                non_number_idxes.append(idx)
        strin += ");"

        # convert arbitrary objects to strings
        for row in range(len(value_list)):
            for col in non_number_idxes:
                value_list[row][col] = str(value_list[row][col])

        query = " ".join(
            ["INSERT INTO", table_name, str(tuple(keys)).replace("'", ""), "VALUES", strin])

        sql_run.executemany(query, value_list)
        conn.commit()
        conn.close()
