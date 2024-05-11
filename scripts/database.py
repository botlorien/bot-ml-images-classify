import logging
import psycopg2
import sqlite3
import re, os, time
from datetime import datetime
import customtkinter
import string
from cryptography.fernet import Fernet
from PIL import Image
import scripts.globals as gb

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

class PostgresqlConnection:

    def __init__(self, database):
        """
        Initialize a PostgreSQL database connection.

        :param database: The name of the database.
        :type database: str
        """
        self.ui = DatabaseInterface(database)
        self.credentials = None
        self.conn = None
        self.cursor = None
        self._load_credentials()

    def _load_credentials(self):
        """
        Loads the user's credentials from the interface or asks for them if they haven't been set yet.
        """
        self.credentials = self.ui.load_credentials()
        if self.credentials is None:
            self.ui.ask_credentials()
            self.credentials = self.ui.load_credentials()

    def connect(self, encoding: str = ''):
        """
        Connect to the PostgreSQL database using the loaded credentials.
        :param encoding: The encoding to setup on the connection.
        :type encoding: str
        """
        self.conn = psycopg2.connect(
            host=self.credentials[0],  # or the IP address of your postgres server
            database=self.credentials[1],
            user=self.credentials[2],
            password=self.credentials[3],
            sslmode=self.credentials[4],
            port=self.credentials[5],  # or the port your postgres server is running on

        )
        if encoding != '':
            self.conn.set_client_encoding(encoding)

        self.cursor = self.conn.cursor()
        self.cursor.execute('BEGIN TRANSACTION')

    def close(self):
        """
        Close the database connection.
        """
        if self.conn:
            self.cursor.execute('END TRANSACTION')
            self.cursor.close()
            self.conn.close()

    class ConnectionFail(Exception):
        pass

    def test_conection(self):
        self.connect()
        self.close()

    def handle_connection(self, func, *args, **kwargs):
        """
        Handle the PostgreSQL database connection and execute a function.

        This method establishes a connection to the PostgreSQL database, executes the specified function,
        commits the changes, and handles any errors that occur during the connection.

        :param func: The function to execute.
        :type func: function
        :param args: Positional arguments to pass to the function.
        :param kwargs: Keyword arguments to pass to the function.
        :param time_out: The timeout duration for establishing the connection, defaults to 20 seconds.
        :type time_out: int, optional
        :param raise_exception: Whether to raise an exception on connection failure, defaults to True.
        :type raise_exception: bool, optional
        :return: The result of the executed function.
        :rtype: any
        :raises ConnectionFail: If the connection fails and raise_exception is True.
        """
        time_out = 20 if 'time_out' not in kwargs else kwargs.pop('time_out')
        raise_exception = True if 'raise_exception' not in kwargs else kwargs.pop('raise_exception')
        contador_time_out = 0
        ret = False
        while contador_time_out < time_out:
            try:
                encoding = kwargs.pop('encoding') if 'encoding' in kwargs else ''
                self.connect(encoding)
                ret = func(*args, **kwargs)
                self.conn.commit()
                break
            except (Exception, psycopg2.Error) as error:
                logging.exception(error)
                self.cursor.execute("ROLLBACK;")
                time.sleep(1)
            finally:
                self.close()
            contador_time_out += 1
        if contador_time_out >= time_out and raise_exception:
            raise PostgresqlConnection.ConnectionFail
        return ret


class Postgresql(PostgresqlConnection):

    def __init__(self, database: str = 'botlorien'):
        """
        Initialize a PostgreSQL connection.

        :param database: The name of the database.
        :type database: str
        """
        super().__init__(database)

        self.time_out_conection = None

    def _show_properties_database(self):
        """
        Print the properties of the PostgreSQL database connection.
        """
        print(self.conn.get_dsn_parameters(), "\n")

        # Print PostgreSQL version
        self.cursor.execute("SELECT version();")
        record = self.cursor.fetchone()
        print("You are connected to - ", record, "\n")

    def show_properties_database(self):
        """
        Show the properties of the PostgreSQL database connection.
        """
        self.handle_connection(self._show_properties_database)

    def set_time_out_postgres_options(self, time_out_conection):
        """
        Set the timeout option for the PostgreSQL connection.

        :param time_out_conection: The timeout duration in seconds.
        :type time_out_conection: int
        """
        self.time_out_conection = time_out_conection
        os.environ['PGOPTIONS'] = f'-c statement_timeout={self.time_out_conection}s'

    def _create_table(self, table_name, columns):
        """
        Create a table in the PostgreSQL database.

        :param table_name: The name of the table.
        :type table_name: str
        :param columns: The columns to create in the table.
        :type columns: str
        """
        sql_comand = f"""CREATE TABLE IF NOT EXISTS {table_name} ({columns});"""
        self.cursor.execute(sql_comand)

    def create_table(self, table_name, columns):
        """
        Create a table in the PostgreSQL database.

        :param table_name: The name of the table.
        :type table_name: str
        :param columns: The columns to create in the table.
        :type columns: str
        """
        self.handle_connection(self._create_table, table_name, columns)

    def _create_column(self, table_name, column, type_column):
        """
        Add a column to an existing table in the PostgreSQL database.

        :param table_name: The name of the table.
        :type table_name: str
        :param column: The name of the column to create.
        :type column: str
        :param type_column: The data type of the column.
        :type type_column: str
        """
        sql_create_column = f'ALTER TABLE {table_name} ADD IF NOT EXISTS {column} {type_column};'
        self.cursor.execute(sql_create_column)

    def create_column(self, table_name, column, type_column):
        """
        Add a column to an existing table in the PostgreSQL database.

        :param table_name: The name of the table.
        :type table_name: str
        :param column: The name of the column to create.
        :type column: str
        :param type_column: The data type of the column.
        :type type_column: str
        """
        self.handle_connection(self._create_column, table_name, column, type_column)

    def clear_table_columns_to_db(self, colunas):
        """
        Clear the table columns for the PostgreSQL database.

        :param colunas: The column names to clear.
        :type colunas: list
        :return: The cleared column names.
        :rtype: list
        """
        colunas = [re.sub(r'\W+', '_', coluna) for coluna in
                   colunas]
        return colunas

    def _define_column_object_type(self, table, column):
        """
        Define the object type of a column in the PostgreSQL database.

        :param table: The table containing the column.
        :type table: pandas.DataFrame
        :param column: The name of the column.
        :type column: str
        :return: The PostgreSQL data type for the column.
        :rtype: str
        """
        # get the maximum length of the col column
        max_len = table[column].astype('str').str.len().max()

        # get the minimun length of the col column
        min_len = table[column].astype('str').str.len().min()

        # if len values is the same define varchar type
        if max_len == min_len:
            postgres_type = f'VARCHAR({max_len})'
        else:
            # else text type
            postgres_type = 'TEXT'
        return postgres_type

    def generate_columns_types(self, table, dtypes_columns=None, match_partial: bool = False, id_column: bool = True):
        """
        Generate the column types for a PostgreSQL table based on a Pandas DataFrame.


        :param table: The DataFrame containing the table data.
        :type table: pandas.DataFrame
        :param dtypes_columns: Optional dictionary specifying custom column types for specific columns.
        :type dtypes_columns: dict, optional
        :param match_partial: A flag to enable search partial name columns to match dtypes_columns keys.
        :type match_partial: bool, optional
        :param id_column: A flag to create the column id.
        :type id_column: bool, optional
        :return: The PostgreSQL column definitions.
        :rtype: str
        """
        if dtypes_columns is None:
            dtypes_columns = {}

        # Define a mapping of Pandas data types to PostgreSQL data types
        type_map = {'int64': 'NUMERIC', 'float64': 'NUMERIC', 'object': 'TEXT', 'datetime64[ns]': 'TIMESTAMP'}

        # Loop through each column in the DataFrame and generate a PostgreSQL column definition
        postgres_columns = ['id serial PRIMARY KEY'] if id_column else []

        for column in table.columns:
            dtype = str(table[column].dtype)

            if len(dtypes_columns) > 0:
                matched_column = [key for key in dtypes_columns.keys() if key.lower() in column.lower()]
                if (match_partial and len(matched_column) > 0) \
                        or column in dtypes_columns.keys():
                    postgres_columns.append(f'"{column}" {dtypes_columns[matched_column[0]]}')
                    continue

            # Verificando tipo objeto
            if dtype == 'object':
                postgres_type = self._define_column_object_type(table, column)
            else:
                postgres_type = type_map.get(dtype, 'TEXT')

            postgres_columns.append(f'"{column}" {postgres_type}')

        # Join the list of column definitions into a single string with commas between them
        postgres_columns_str = ', '.join(postgres_columns)
        print(postgres_columns_str)

        return postgres_columns_str

    def _insert_table_db(self, table, table_name, columns_postgre):
        """
        Insert a table into a PostgreSQL database.

        :param table: The DataFrame containing the table data.
        :type table: pandas.DataFrame
        :param table_name: The name of the table in the database.
        :type table_name: str
        :param columns_postgre: The column names in the PostgreSQL table.
        :type columns_postgre: str
        """
        # define a list of '%s' in the range columns
        lista_var_linha = ['%s' for _ in table.columns]

        # turn the list into a string
        var_linha = ' ,'.join(lista_var_linha)

        # generate a list o list query values from the table to be inserted on database
        table_list = list(
            map(lambda lista: tuple(list(
                map(lambda vl: str(vl).replace("'", "''") if (str(vl) != 'NULL') else None, lista))),
                table.values.tolist()))

        # SQL comand to insert many
        sql_insert = f"INSERT INTO {table_name}({columns_postgre}) VALUES ({var_linha});"
        self.cursor.executemany(sql_insert, table_list)

    def insert_table_db(self, table, table_name, columns_postgre):
        """
        Insert a table into a PostgreSQL database.

        :param table: The DataFrame containing the table data.
        :type table: pandas.DataFrame
        :param table_name: The name of the table in the database.
        :type table_name: str
        :param columns_postgre: The column names in the PostgreSQL table.
        :type columns_postgre: str
        """
        self.handle_connection(self._insert_table_db, table, table_name, columns_postgre)

    def to_postgresql(self, table, table_name: str = '', call_procedure: str = '', dtypes_columns=None,
                      match_partial: bool = False, id_column: bool = True):
        """
        Export a table to a PostgreSQL database.

        :param table: The DataFrame containing the table data.
        :type table: pandas.DataFrame
        :param table_name: The name of the table in the database. If not provided, the table name will be generated automatically.
        :type table_name: str
        :param call_procedure: The name of the stored procedure to call after inserting the table data (optional).
        :type call_procedure: str
        :param dtypes_columns: A dictionary specifying the desired data types for specific columns (optional).
        :type dtypes_columns: dict
        :param id_column: A flag to create the column id.
        :type id_column: bool, optional
        :return: True if the export was successful, False otherwise.
        :rtype: bool
        """
        postgres_columns = self.generate_columns_types(table, dtypes_columns, match_partial, id_column)
        self.create_table(table_name, postgres_columns)
        postgres_columns = self.select_columns(table_name, id_column)
        self.insert_table_db(table, table_name, postgres_columns)

        if len(call_procedure) > 0:
            self.call_procedure(call_procedure)
        return True

    def _call_procedure(self, procedure_name):
        """
        Call a stored procedure in the PostgreSQL database.

        :param procedure_name: The name of the stored procedure.
        :type procedure_name: str
        """
        sql_call_procedure = f'CALL {procedure_name}()'
        self.cursor.execute(sql_call_procedure)

    def call_procedure(self, procedure_name):
        """
        Call a stored procedure in the PostgreSQL database.

        :param procedure_name: The name of the stored procedure.
        :type procedure_name: str
        """
        self.handle_connection(self._call_procedure, procedure_name)

    def _select_columns(self, table_name, ignore_id: bool = True):
        """
        Retrieve the column names of a table in the PostgreSQL database.

        :param table_name: The name of the table.
        :type table_name: str
        :return: The column names of the table.
        :rtype: str
        """
        sql_select_column = f"""
            select "column_name"
            from INFORMATION_SCHEMA.COLUMNS
            where TABLE_NAME='{table_name}'
            ORDER BY ordinal_position
        """
        self.cursor.execute(sql_select_column)
        tabela_columns = self.cursor.fetchall()

        colnames = [f'"{desc[0]}"' for desc in tabela_columns]

        if ignore_id:
            colnames = colnames[1:]

        colunas_postgre = ' ,'.join(colnames)

        return colunas_postgre

    def select_columns(self, table_name, ignore_id: bool = True):
        """
        Retrieve the column names of a table in the PostgreSQL database.

        :param table_name: The name of the table.
        :type table_name: str
        :param ignore_id: If you want ignore the id column.
        :type ignore_id: bool
        :return: The column names of the table.
        :rtype: str
        """
        colunas_postgre = self.handle_connection(self._select_columns, table_name, ignore_id)
        return colunas_postgre

    def _execute_script(self, script):
        """
        Execute a SQL script in the PostgreSQL database.

        :param script: The SQL script to execute.
        :type script: str
        :return: The result of the script execution.
        :rtype: list
        """
        self.cursor.execute(script)
        if 'select' in script.lower():
            return self.cursor.fetchall()

    def execute_script(self, script, **kwargs):
        """
        Execute a SQL script in the PostgreSQL database.

        :param script: The SQL script to execute.
        :type script: str
        :return: The result of the script execution.
        :rtype: list
        """
        return self.handle_connection(self._execute_script, script, **kwargs)


class SqliteConnection:

    def __init__(self, database):
        """
        Initialize a Sqlite database connection.

        :param database: The name of the database.
        :type database: str
        """
        self.database = database
        self.conn = None
        self.cursor = None

    def connect(self):
        """
        Connect to the PostgreSQL database using the loaded credentials.
        """
        self.conn = sqlite3.connect(f'{self.database}.db')

        self.cursor = self.conn.cursor()

    def close(self):
        """
        Close the database connection.
        """
        if self.conn:
            self.cursor.close()
            self.conn.close()

    class ConnectionFail(Exception):
        pass

    def handle_connection(self, func, *args, **kwargs):
        """
        Handle the Sqlite database connection and execute a function.

        This method establishes a connection to the Sqlite database, executes the specified function,
        commits the changes, and handles any errors that occur during the connection.

        :param func: The function to execute.
        :type func: function
        :param args: Positional arguments to pass to the function.
        :param kwargs: Keyword arguments to pass to the function.
        :param time_out: The timeout duration for establishing the connection, defaults to 20 seconds.
        :type time_out: int, optional
        :param raise_exception: Whether to raise an exception on connection failure, defaults to True.
        :type raise_exception: bool, optional
        :return: The result of the executed function.
        :rtype: any
        :raises ConnectionFail: If the connection fails and raise_exception is True.
        """
        time_out = 20 if 'time_out' not in kwargs else kwargs.pop('time_out')
        raise_exception = True if 'raise_exception' not in kwargs else kwargs.pop('raise_exception')
        contador_time_out = 0
        ret = False
        while contador_time_out < time_out:
            try:
                self.connect()
                ret = func(*args, **kwargs)
                self.conn.commit()
                break
            except Exception as error:
                print("Error while connecting to Sqlite", error)
                time.sleep(1)
            finally:
                self.close()
            contador_time_out += 1
        if contador_time_out >= time_out and raise_exception:
            raise SqliteConnection.ConnectionFail
        return ret


class Sqlite(SqliteConnection):

    def __init__(self, database: str = 'botlorien'):
        """
        Initialize a Sqlite connection.

        :param database: The name of the database.
        :type database: str
        """
        super().__init__(database)

    def _show_properties_database(self):
        """
        Print the properties of the Sqlite database connection.
        """

        # Print Sqlite version
        self.cursor.execute("SELECT sqlite_version();")
        record = self.cursor.fetchone()
        print("You are connected to - ", record, "\n")

    def show_properties_database(self):
        """
        Show the properties of the Sqlite database connection.
        """
        self.handle_connection(self._show_properties_database)

    def _create_table(self, table_name, columns):
        """
        Create a table in the Sqlite database.

        :param table_name: The name of the table.
        :type table_name: str
        :param columns: The columns to create in the table.
        :type columns: str
        """
        sql_comand = f"""CREATE TABLE IF NOT EXISTS {table_name} ({columns});"""
        self.cursor.execute(sql_comand)

    def create_table(self, table_name, columns):
        """
        Create a table in the Sqlite database.

        :param table_name: The name of the table.
        :type table_name: str
        :param columns: The columns to create in the table.
        :type columns: str
        """
        self.handle_connection(self._create_table, table_name, columns)

    def _create_column(self, table_name, column, type_column):
        """
        Add a column to an existing table in the Sqlite database.

        :param table_name: The name of the table.
        :type table_name: str
        :param column: The name of the column to create.
        :type column: str
        :param type_column: The data type of the column.
        :type type_column: str
        """
        sql_create_column = f'ALTER TABLE {table_name} ADD IF NOT EXISTS {column} {type_column};'
        self.cursor.execute(sql_create_column)

    def create_column(self, table_name, column, type_column):
        """
        Add a column to an existing table in the Sqlite database.

        :param table_name: The name of the table.
        :type table_name: str
        :param column: The name of the column to create.
        :type column: str
        :param type_column: The data type of the column.
        :type type_column: str
        """
        self.handle_connection(self._create_column, table_name, column, type_column)

    def clear_table_columns_to_db(self, colunas):
        """
        Clear the table columns for the Sqlite database.

        :param colunas: The column names to clear.
        :type colunas: list
        :return: The cleared column names.
        :rtype: list
        """
        colunas = [re.sub(r'\W+', '_', coluna) for coluna in
                   colunas]
        return colunas

    def _define_column_object_type(self, table, column):
        """
        Define the object type of a column in the Sqlite database.

        :param table: The table containing the column.
        :type table: pandas.DataFrame
        :param column: The name of the column.
        :type column: str
        :return: The Sqlite data type for the column.
        :rtype: str
        """
        # get the maximum length of the col column
        max_len = table[column].astype('str').str.len().max()

        # get the minimun length of the col column
        min_len = table[column].astype('str').str.len().min()

        # if len values is the same define varchar type
        if max_len == min_len:
            postgres_type = f'VARCHAR({max_len})'
        else:
            # else text type
            postgres_type = 'TEXT'
        return postgres_type

    def generate_columns_types(self, table, dtypes_columns=None):
        """
        Generate the column types for a Sqlite table based on a Pandas DataFrame.

        :param table: The DataFrame containing the table data.
        :type table: pandas.DataFrame
        :param dtypes_columns: Optional dictionary specifying custom column types for specific columns.
        :type dtypes_columns: dict, optional
        :return: The Sqlite column definitions.
        :rtype: str
        """
        if dtypes_columns is None:
            dtypes_columns = {}

        # Define a mapping of Pandas data types to Sqlite data types
        type_map = {'int64': 'NUMERIC', 'float64': 'NUMERIC', 'object': 'TEXT', 'datetime64[ns]': 'TIMESTAMP'}

        # Loop through each column in the DataFrame and generate a Sqlite column definition
        postgres_columns = ['id INTEGER PRIMARY KEY AUTOINCREMENT']

        for column in table.columns:
            dtype = str(table[column].dtype)

            if len(dtypes_columns) > 0 and column in dtypes_columns.keys():
                postgres_columns.append(f'"{column}" {dtypes_columns[column]}')
                continue

            # Verificando tipo objeto
            if dtype == 'object':
                postgres_type = self._define_column_object_type(table, column)
            else:
                postgres_type = type_map.get(dtype, 'TEXT')

            postgres_columns.append(f'"{column}" {postgres_type}')

        # Join the list of column definitions into a single string with commas between them
        postgres_columns_str = ', '.join(postgres_columns)

        return postgres_columns_str

    def _insert_table_db(self, table, table_name, columns_postgre):
        """
        Insert a table into a Sqlite database.

        :param table: The DataFrame containing the table data.
        :type table: pandas.DataFrame
        :param table_name: The name of the table in the database.
        :type table_name: str
        :param columns_postgre: The column names in the Sqlite table.
        :type columns_postgre: str
        """
        # define a list of '?' in the range columns
        lista_var_linha = ['?' for _ in table.columns]

        # turn the list into a string
        var_linha = ' ,'.join(lista_var_linha)

        # generate a list o list query values from the table to be inserted on database
        table_list = list(
            map(lambda lista: tuple(list(
                map(lambda vl: str(vl).replace("'", "''") if (str(vl) != 'NULL') else None, lista))),
                table.values.tolist()))

        # SQL comand to insert many
        sql_insert = f"INSERT INTO {table_name}({columns_postgre}) VALUES ({var_linha});"

        self.cursor.executemany(sql_insert, table_list)

    def insert_table_db(self, table, table_name, columns_postgre):
        """
        Insert a table into a Sqlite database.

        :param table: The DataFrame containing the table data.
        :type table: pandas.DataFrame
        :param table_name: The name of the table in the database.
        :type table_name: str
        :param columns_postgre: The column names in the Sqlite table.
        :type columns_postgre: str
        """
        self.handle_connection(self._insert_table_db, table, table_name, columns_postgre)

    def to_sqlite(self, table, table_name: str = '', dtypes_columns=None):
        """
        Export a table to a Sqlite database.

        :param table: The DataFrame containing the table data.
        :type table: pandas.DataFrame
        :param table_name: The name of the table in the database. If not provided, the table name will be generated automatically.
        :type table_name: str
        :param dtypes_columns: A dictionary specifying the desired data types for specific columns (optional).
        :type dtypes_columns: dict
        :return: True if the export was successful, False otherwise.
        :rtype: bool
        """
        colunas_sqlite = self.generate_columns_types(table, dtypes_columns)
        self.create_table(table_name, colunas_sqlite)
        colunas_sqlite = self.select_columns(table_name)
        self.insert_table_db(table, table_name, colunas_sqlite)
        return True

    def _select_columns(self, table_name):
        """
        Retrieve the column names of a table in the Sqlite database.

        :param table_name: The name of the table.
        :type table_name: str
        :return: The column names of the table.
        :rtype: str
        """
        sql_select_column = f"""
            PRAGMA table_info({table_name})
        """
        self.cursor.execute(sql_select_column)
        tabela_columns = self.cursor.fetchall()

        colnames = [f'"{desc[1]}"' for desc in tabela_columns][1:]

        colunas_sqlite = ' ,'.join(colnames)
        print(colunas_sqlite)

        return colunas_sqlite

    def select_columns(self, table_name):
        """
        Retrieve the column names of a table in the Sqlite database.

        :param table_name: The name of the table.
        :type table_name: str
        :return: The column names of the table.
        :rtype: str
        """
        colunas_sqlite = self.handle_connection(self._select_columns, table_name)
        return colunas_sqlite

    def _execute_script(self, script):
        """
        Execute a SQL script in the Sqlite database.

        :param script: The SQL script to execute.
        :type script: str
        :return: The result of the script execution.
        :rtype: list
        """
        self.cursor.execute(script)
        return self.cursor.fetchall()

    def execute_script(self, script):
        """
        Execute a SQL script in the Sqlite database.

        :param script: The SQL script to execute.
        :type script: str
        :return: The result of the script execution.
        :rtype: list
        """
        return self.handle_connection(self._execute_script, script)


class DatabaseInterface():

    def __init__(self, database: str = 'botlorien'):
        """
        Constructor method.
        Initializes the DatabaseInterface object with several tkinter widgets and an instance of the DataBaseCript class.
        """

        # configurando to titulo do app e a geometria dinamica
        self.database = database
        gb.init_root().title(f"{database} DataBase")
        gb.init_root().geometry("+100+100")

        self.entry_frame = None
        self.entry_login = None
        self.label_login = None
        self.button_login = None
        self.credentials = None
        self.messagebox_window = None
        self.label_image = None
        self.label_messabox = None
        self.button_messagebox = None
        self.hided = False

        self.path_credentials = os.path.join(os.getcwd(), f'parameters/{database.lower()}_credentials.bot')

        self.cript = DataBaseCript()

    @staticmethod
    def init_root():
        global ROOT
        try:
            customtkinter.CTkFrame(ROOT)
        except Exception as e:
            ROOT = customtkinter.CTk()
            return ROOT
        else:
            return ROOT

    def _create_folder(self, path):
        """
        Creates a folder at the specified path if it does not already exist.

        :param path: Path to the folder to be created.
        :type path: str
        :return: Absolute path to the folder.
        :rtype: str
        """
        if ':' not in path:
            root_path = os.getcwd()
            path = os.path.abspath(root_path + f'/{path}')

        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def start_root(self):
        self.hide_window()
        gb.init_root().mainloop()

    def ask_credentials(self):
        """
        Presents an interface for entering credentials.

        :return: None
        """
        # criando a interface the inserção das credenciais
        self.entry_frame = customtkinter.CTkFrame(gb.init_root())
        self.entry_frame.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        img_path = os.path.join(PARENT_DIR,'assets/database.png')
        image = self._load_image(img_path, 30, 30)
        self.label_login = customtkinter.CTkLabel(self.entry_frame, text="Insira suas credenciais!",
                                                  font=customtkinter.CTkFont(size=18, weight="bold"),
                                                  image=image,
                                                  compound='left'
                                                  )
        self.label_login.grid(row=1, column=0, padx=(20, 20), pady=(20, 10))

        self.entry_login = [customtkinter.CTkEntry(self.entry_frame) for _ in range(6)]
        self.entry_login[0] = customtkinter.CTkEntry(self.entry_frame, placeholder_text="host", width=70)
        self.entry_login[0].grid(row=2, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.entry_login[1] = customtkinter.CTkEntry(self.entry_frame, placeholder_text="dbname")
        self.entry_login[1].grid(row=3, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.entry_login[2] = customtkinter.CTkEntry(self.entry_frame, placeholder_text="user")
        self.entry_login[2].grid(row=4, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.entry_login[3] = customtkinter.CTkEntry(self.entry_frame, placeholder_text="password", show="*")
        self.entry_login[3].grid(row=5, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.entry_login[4] = customtkinter.CTkEntry(self.entry_frame, placeholder_text="sslmode")
        self.entry_login[4].grid(row=6, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.entry_login[5] = customtkinter.CTkEntry(self.entry_frame, placeholder_text="port")
        self.entry_login[5].grid(row=7, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.button_login = customtkinter.CTkButton(self.entry_frame, text='Avançar', command=self._button_login_event)
        self.button_login.grid(row=8, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        gb.init_root().mainloop()

    def _button_login_event(self):
        """
        Encrypts the entered credentials and stores them if they are valid.

        :return: None
        """
        self.credentials = [self.cript.cripto_data(str(entry.get())) if len(entry.get()) > 0 else None for entry in
                            self.entry_login]
        if None in self.credentials:
            self.messagebox('Insira sua credenciais\n corretamente!')
        else:
            self._store_credentials()

        self.hide_window()
        try:
            self.quit()
        except Exception as e:
            if 'application has been destroyed' not in str(e):
                raise e

    def _store_credentials(self):
        """
        Stores the encrypted credentials in a file.

        :return: None
        """
        self._create_folder('parameters')
        self.credentials = ' ;'.join(self.credentials)
        with open(self.path_credentials, 'w') as file:
            file.write(self.credentials)

    def messagebox(self, message, type_alert: str = 'warning'):
        """
        Presents a messagebox with a given message and alert type.

        :param message: Message to be displayed.
        :type message: str
        :param type_alert: Alert type (either 'warning' or 'info').
        :type type_alert: str
        :return: None
        """
        self.messagebox_window = customtkinter.CTkToplevel(gb.init_root())
        self.messagebox_window.title(type_alert.upper())
        self.messagebox_window.attributes("-topmost", True)

        self.messagebox_window.geometry("300x300+170+170")
        self.messagebox_window.rowconfigure(1, weight=1)
        self.messagebox_window.columnconfigure(0, weight=1)
        img_path_warning = os.path.join(PARENT_DIR, 'assets/warning.png')
        img_path_info = os.path.join(PARENT_DIR, 'assets/info.png')
        image = self._load_image(img_path_warning) if type_alert == 'warning' else self._load_image(
            img_path_info)
        self.label_image = customtkinter.CTkLabel(self.messagebox_window,
                                                  text='',
                                                  image=image,
                                                  compound='left')
        self.label_image.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")

        self.label_messabox = customtkinter.CTkLabel(self.messagebox_window, text=message,
                                                     font=customtkinter.CTkFont(size=20, weight="bold"))
        self.label_messabox.grid(row=1, column=0, padx=20, pady=(20, 10), sticky="nsew")

        self.button_messagebox = customtkinter.CTkButton(self.messagebox_window, text='Ok',
                                                         command=self.messagebox_window.destroy)
        self.button_messagebox.grid(row=2, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.messagebox_window.mainloop()

    @staticmethod
    def _load_image(path, width: int = 50, height: int = 50):
        """
        Loads an image from a specified path and resizes it to a specified width and height.

        :param path_image: Path to the image file.
        :type path_image: str
        :param width: Width to resize the image to.
        :type width: int
        :param height: Height to resize the image to.
        :type height: int
        :return: Resized image.
        :rtype: customtkinter.CTkImage
        """
        # load and create background image
        img = Image.open(path).copy()
        return customtkinter.CTkImage(img, size=(width, height))

    def load_credentials(self):
        """
        Loads the encrypted credentials from a file and decrypts them.

        :return: Decrypted credentials.
        :rtype: list of str
        """
        if os.path.exists(self.path_credentials):
            with open(self.path_credentials, 'r') as file:
                self.credentials = file.read().split(';')
            for i, param in enumerate(self.credentials):
                self.credentials[i] = self.cript.decripto_data(param)
            return self.credentials
        else:
            return None

    def hide_window(self):
        """
        Hides the main tkinter window.

        :return: None
        """
        self.hided = True
        gb.init_root().withdraw()

    def show_window(self):
        """
        Shows the main tkinter window.

        :return: None
        """
        gb.init_root().deiconify()

    def quit(self):
        gb.init_root().quit()
        gb.init_root().destroy()

    def loop(self):
        """
        Starts the tkinter event loop.

        :return: None
        """
        gb.init_root().mainloop()


class DataBaseCript:

    def __init__(self):
        """
        Constructor method.
        Initializes the DataBaseCript object with None as the key.
        """
        self.key = None

    @staticmethod
    def create_folder(path):
        """
        Creates a folder at the specified path.

        :param path: Path to the folder to be created.
        :type path: str
        :return: Absolute path to the folder.
        :rtype: str
        """
        if ':' not in path:
            root_path = os.getcwd()
            path = os.path.abspath(root_path + f'/{path}')

        if not os.path.exists(path):
            os.mkdir(path)
        return path

    @staticmethod
    def create_path_to_save():
        """
        Creates a path for storing the key file.

        :return: Path to the key file.
        :rtype: str
        """
        appdata_path = os.getenv('APPDATA')
        appdata_dir = os.path.dirname(appdata_path)
        path_store_key = os.path.abspath(appdata_dir + '/Local/rpa')
        return path_store_key

    def generate_secret_key(self):
        """
        Generates a Fernet key and saves it into a file.

        :return: None
        """
        key = Fernet.generate_key()
        path_store_key = self.create_path_to_save()
        self.create_folder(path_store_key)
        if not os.path.exists(f"{path_store_key}/h.b"):
            with open(f"{path_store_key}/h.b", "wb") as key_file:
                key_file.write(key)

    def load_key(self):
        """
        Loads the Fernet key from a file.

        :return: None
        """
        path_store_key = self.create_path_to_save()
        self.create_folder(path_store_key)
        with open(f"{path_store_key}/h.b", "rb") as key_file:
            self.key = key_file.read()

    @staticmethod
    def salting(dados):
        """
        Salts the input data by interspersing it with ASCII characters.

        :param dados: Data to be salted.
        :type dados: str
        :return: Salted data.
        :rtype: str
        """
        salted = ''
        salt = string.ascii_uppercase + string.ascii_lowercase
        for i, c in enumerate(dados[::-1]):
            index = salt.find(c)
            salted += c + salt[i] if index == -1 else salt[index - 7] + salt[i]
        return salted

    @staticmethod
    def d_salt(salted):
        """
        Removes the salt from the input data.

        :param salted: Salted data.
        :type salted: str
        :return: Original unsalted data.
        :rtype: str
        """
        original_dados = ''
        salt = string.ascii_uppercase + string.ascii_lowercase
        for i, c in enumerate(salted[::-1]):
            index = salt.find(c)
            index_char_alt_range = salt[-7:].find(c)
            d_salted = salt[index + 7] if index_char_alt_range == -1 else salt[index_char_alt_range]
            if i % 2 != 0:
                original_dados += c if index == -1 else d_salted
        return original_dados

    def cripto_data(self, dados):
        """
        Encrypts and salts the input data.

        :param dados: Data to be encrypted.
        :type dados: str
        :return: Encrypted data.
        :rtype: str
        """
        self.generate_secret_key()
        if self.key is None:
            self.load_key()

        salted = self.salting(dados)
        salted = salted.encode()
        cipher_suite = Fernet(self.key)
        cipher_text = cipher_suite.encrypt(salted)
        dado_text = str(cipher_text.decode())
        return dado_text

    def decripto_data(self, dado_text):
        """
        Decrypts the input data and removes the salt.

        :param dado_text: Data to be decrypted.
        :type dado_text: str
        :return: Decrypted data.
        :rtype: str
        """
        if self.key is None:
            self.load_key()
        cipher_suite = Fernet(self.key)
        plain_text = cipher_suite.decrypt(dado_text.encode())
        salted = str(plain_text.decode())
        original_dado = self.d_salt(salted)
        return original_dado


class BotsDatabase:

    def __init__(self, database: str = 'botlorien'):
        """
        Initialize the BotsDatabase object.

        :param database: The name of the database.
        :type database: str
        """
        self.pg = PostgresqlConnection(database)
        self.verify_credentials(self.pg.credentials, database)
        self.sqlite = SqliteConnection(database)

    class ValuesNotInitiated(Exception):
        pass

    class WrongBotDatabase(Exception):
        pass

    def verify_credentials(self, credentials, database):
        """
        Verify the credentials for accessing the database.

        :param credentials: The database credentials.
        :type credentials: tuple
        :param database: The name of the database.
        :type database: str
        """
        invalid_authentication = False
        try:
            self.pg.test_conection()
        except Exception as e:
            logging.exception(e)
            invalid_authentication = True
        # if credentials[0] not in ('127.0.0.1', 'localhost') or credentials[
        #     1] != database or invalid_authentication:
        #     os.remove(f'parameters/{database.lower()}_credentials.bot')
        #     raise BotsDatabase.WrongBotDatabase


class TableBots(BotsDatabase):

    def __init__(self, ):
        """
        Initialize the TableBots object.

        """
        super().__init__()
        self.bot_id = None
        self.bot_name = None
        self.bot_description = None
        self.initiated_values = False
        self.create_table_if_not_exists()

    def init_values(self, bot_name, bot_description):
        """
        Initialize the values of the TableBots object.

        :param bot_name: The name of the bot.
        :type bot_name: str
        :param bot_description: The description of the bot.
        :type bot_description: str
        """
        self.bot_name = bot_name
        self.bot_description = bot_description
        self.bot_id = self.generate_bot_id(self.bot_name)
        self.initiated_values = True

    def show_values(self):
        """
        Show the initialized values.

        """
        if self.initiated_values:
            print(f"""
                bot_id:{self.bot_id},
                bot_name:{self.bot_name},
                bot_description:{self.bot_description},
            """)
        else:
            raise self.ValuesNotInitiated

    def _create_table_if_not_exists(self):
        """
        Create the bots table if it does not exist.

        """
        # Creating the Bots Table
        self.pg.cursor.execute("""
        CREATE TABLE IF NOT EXISTS bots(
            bot_id NUMERIC PRIMARY KEY,
            bot_name TEXT NOT NULL,
            bot_description TEXT,
            bot_created_date TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """)

    def create_table_if_not_exists(self):
        """
        Create the bots table if it does not exist.

        """
        self.pg.handle_connection(self._create_table_if_not_exists)

    def _insert_or_update(self):
        """
        Insert or update a bot when it is registered.

        """
        self.pg.cursor.execute(f"""
        INSERT INTO bots (bot_id, bot_name, bot_description)
        VALUES ({self.bot_id}, '{self.bot_name}', '{self.bot_description}')
        ON CONFLICT (bot_id) 
        DO UPDATE SET bot_description = EXCLUDED.bot_description;
        """)

    def register_bot(self):
        """
        Register the bot.

        """
        if self.initiated_values:
            self.pg.handle_connection(self._insert_or_update)
        else:
            raise self.ValuesNotInitiated

    @staticmethod
    def generate_bot_id(bot_name):
        """
        Generate a unique bot ID based on the bot name.

        :param bot_name: The name of the bot.
        :type bot_name: str
        :return: The generated bot ID.
        :rtype: int
        """
        st = string.ascii_uppercase + string.ascii_lowercase
        bot_id = ''
        for i, c in enumerate(bot_name):
            index = st.find(c)
            bot_id += str(index) if index != -1 else c
        return int(bot_id)


class TaskTable(BotsDatabase):

    def __init__(self):
        """
        Initialize the TaskTable object.

        """
        super().__init__()
        self.task_id = None
        self.bot_id = None
        self.task_name = None
        self.task_description = None
        self.pc_path = None
        self.user_login = None
        self.bot_version = None
        self.task_end_time = None
        self.initiated_values = True
        self.create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        """
        Create the tasks table if it does not exist.

        """
        # Creating the Tasks Table
        self.pg.cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks(
            task_id NUMERIC PRIMARY KEY,
            bot_id NUMERIC NOT NULL,
            task_name TEXT NOT NULL,
            task_description TEXT,
            FOREIGN KEY (bot_id) REFERENCES bots (bot_id)
        );
        """)

    def init_values(self, bot_id, task_name, task_description):
        """
        Initialize the values of the TaskTable object.

        :param bot_id: The bot ID.
        :type bot_id: int
        :param task_name: The task name.
        :type task_name: str
        :param task_description: The task description.
        :type task_description: str
        """
        self.task_id = self.generate_task_id(bot_id, task_name)
        self.bot_id = bot_id
        self.task_name = task_name
        self.task_description = task_description
        self.initiated_values = True

    def show_values(self):
        """
        Show the initialized values.

        """
        if self.initiated_values:
            print(f"""
                task_id = {self.task_id}
                bot_id = {self.bot_id}
                task_name = {self.task_name}
                task_description = {self.task_description}

            """)
        else:
            raise self.ValuesNotInitiated

    def create_table_if_not_exists(self):
        """
        Create the tasks table if it does not exist.

        """
        self.pg.handle_connection(self._create_table_if_not_exists)

    def _insert_or_update_on_register_task(self):
        """
        Insert or update a task when it is registered.

        """
        self.pg.cursor.execute(f"""
        INSERT INTO tasks (task_id,bot_id, task_name, task_description)
        VALUES ({self.task_id},{self.bot_id}, '{self.task_name}', '{self.task_description}')
        ON CONFLICT (task_id) 
        DO UPDATE SET task_description = EXCLUDED.task_description;
        """)

    def register_task(self):
        """
        Register the task.

        """
        if self.initiated_values:
            self.pg.handle_connection(self._insert_or_update_on_register_task)
        else:
            raise self.ValuesNotInitiated

    @staticmethod
    def generate_task_id(bot_id, task_name):
        """
        Generate a unique task ID based on the bot ID and task name.

        :param bot_id: The bot ID.
        :type bot_id: int
        :param task_name: The task name.
        :type task_name: str
        :return: The generated task ID.
        :rtype: int
        """
        st = string.ascii_uppercase + string.ascii_lowercase
        task_id = ''
        for i, c in enumerate(task_name):
            index = st.find(c)
            task_id += str(index) if index != -1 else c
        return int(bot_id) + int(task_id)


class TaskResults(BotsDatabase):

    def __init__(self):
        """
        Initialize the TaskResults object.

        """
        super().__init__()
        self.result_id = None
        self.task_id = None
        self.pc_path = None
        self.user_login = None
        self.bot_version = None
        self.result_status = None
        self.result_data = None
        self.error_message = None
        self.initiated_values = False
        self.create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        """
        Create the tasks_results table if it does not exist.

        """
        # Creating the Tasks_Results Table
        self.pg.cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks_results(
            result_id NUMERIC PRIMARY KEY,
            task_id NUMERIC NOT NULL,
            pc_path TEXT,
            user_login TEXT,
            bot_version TEXT,
            result_status TEXT NOT NULL,
            result_data TEXT,
            error_message TEXT,
            task_start_time TIMESTAMP NOT NULL DEFAULT NOW(),
            task_end_time TIMESTAMP DEFAULT NULL,
            duration_task NUMERIC DEFAULT NULL,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id)
        );
        """)

    def init_values(self, task_id, pc_path, user_login, bot_version):
        """
        Initialize the values of the TaskResults object.

        :param task_id: The task ID.
        :type task_id: int
        :param pc_path: The PC path.
        :type pc_path: str
        :param user_login: The user login.
        :type user_login: str
        :param bot_version: The bot version.
        :type bot_version: str
        """
        self.result_id = self.generate_result_id(task_id)
        self.task_id = task_id
        self.pc_path = pc_path
        self.user_login = user_login
        self.bot_version = bot_version
        self.result_status = False
        self.result_data = None
        self.error_message = None
        self.initiated_values = True

    def show_values(self):
        """
        Show the initialized values.

        """
        if self.initiated_values:
            print(f"""
                result_id = {self.result_id}
                task_id = {self.task_id}
                pc_path = {self.pc_path}
                user_login = {self.user_login}
                bot_version = {self.bot_version}
                result_status = {self.result_status}
                result_data = {self.result_data}
                error_message = {self.error_message}
            """)
        else:
            raise self.ValuesNotInitiated

    def create_table_if_not_exists(self):
        """
        Create the tasks_results table if it does not exist.

        """
        self.pg.handle_connection(self._create_table_if_not_exists)

    def _insert_on_open_task(self):
        """
        Insert a new task result when the task is opened.

        """
        self.pg.cursor.execute(f"""
        INSERT INTO tasks_results (result_id,task_id, pc_path, user_login, bot_version, result_status, result_data,error_message)
        VALUES ({self.result_id},{self.task_id}, '{self.pc_path}', '{self.user_login}', '{self.bot_version}',
         '{self.result_status}', '{self.result_data}', '{self.error_message}');
        """)

    def _update_on_close_task_old(self):
        """
        Update the task result when the task is closed.

        """
        self.pg.cursor.execute(f"""
        UPDATE tasks_results
        SET task_end_time = NOW(), result_status = '{self.result_status}', result_data = '{self.result_data}',
        error_message = '{self.error_message}'
        WHERE result_id = {self.result_id};

        UPDATE tasks_results
        SET duration_task = CAST(EXTRACT(EPOCH FROM (task_end_time - task_start_time)) AS INTEGER);
        """)

    def _skip_locked(self):
        """
        Update the task result when the task is closed.

        """
        self.pg.cursor.execute(f"""
        SELECT * FROM tasks_results
        WHERE result_id = {self.result_id}
        FOR UPDATE SKIP LOCKED;
        """)

    def _update_duration_tasks(self):
        """
        Update the duration task for all

        """
        self.pg.cursor.execute("""
        UPDATE tasks_results
        SET duration_task = EXTRACT(EPOCH FROM (task_end_time - task_start_time)),
        WHERE task_start_time IS NOT NULL AND task_end_time IS NOT NULL
        FOR UPDATE SKIP LOCKED;
        """)

    def _update_on_close_task(self):
        """
        Update the task result when the task is closed.

        """
        self._skip_locked()
        self.pg.cursor.execute(f"""
        UPDATE tasks_results
        SET task_end_time = NOW(), result_status = '{self.result_status}', result_data = '{self.result_data}',
        error_message = '{self.error_message}'
        WHERE result_id = {self.result_id};
        """)

        self.pg.cursor.execute(f"""
        UPDATE tasks_results
        SET duration_task = CAST(EXTRACT(EPOCH FROM (task_end_time - task_start_time)) AS INTEGER)
        WHERE result_id = {self.result_id} AND task_start_time IS NOT NULL;
        """)
        # self._update_duration_tasks()

    def open_task(self):
        """
        Open the task.

        """
        if self.initiated_values:
            self.pg.handle_connection(self._insert_on_open_task)
        else:
            raise self.ValuesNotInitiated

    def end_task(self, result_status, result_data, error_message):
        """
        End the task with the given result status, data, and error message.

        :param result_status: The status of the task result.
        :type result_status: bool
        :param result_data: The data of the task result.
        :type result_data: str
        :param error_message: The error message of the task result.
        :type error_message: str
        """
        if self.initiated_values:
            self.result_status = result_status
            self.result_data = result_data
            self.error_message = error_message
            self.pg.handle_connection(self._update_on_close_task)
        else:
            raise self.ValuesNotInitiated

    @staticmethod
    def generate_result_id(task_id):
        """
        Generate a unique result ID based on the task ID and current timestamp.

        :param task_id: The task ID.
        :type task_id: int
        :return: The generated result ID.
        :rtype: int
        """
        now = datetime.now()
        timestamp = now.timestamp()
        return task_id + int(timestamp)


class BotTable:

    def __init__(self):
        """
        Initialize the BotTable object.

        """
        self.table_bot = TableBots()
        self.task_table = TaskTable()
        self.task_result = TaskResults()

    # register bot
    def register_bot(self, bot_name: str, bot_description: str):
        """
        Register a bot with its name and description.

        :param bot_name: The name of the bot.
        :type bot_name: str
        :param bot_description: The description of the bot.
        :type bot_description: str
        """
        self.table_bot.init_values(bot_name, bot_description)
        self.table_bot.register_bot()

    # register task
    def register_task(self, task: str, task_description: str):
        """
        Register a task with its name and description.

        :param task: The name of the task.
        :type task: str
        :param task_description: The description of the task.
        :type task_description: str
        """
        self.task_table.init_values(self.table_bot.bot_id, task, task_description)
        self.task_table.register_task()

    # initializing new result from task
    def create_new_instance_from_task(self, bot_version, user: str = ''):
        """
        Create a new instance for a task result.

        :param bot_version: The version of the bot.
        :type bot_version: str
        :param user: The name of the user.
        :type user: str
        """
        self.task_result.init_values(self.task_table.task_id, os.getcwd(), user, bot_version)

    # opening the new task result
    def open_task(self):
        """
        Open the task result.

        """
        self.task_result.open_task()

    def end_task(self, status: bool, resultado: str, if_erro: str):
        """
        End the task with the given status, result, and error message.

        :param status: The status of the task.
        :type status: bool
        :param resultado: The result of the task.
        :type resultado: str
        :param if_erro: The error message if the task failed.
        :type if_erro: str
        """
        self.task_result.end_task(status, resultado, if_erro)


def migrate_data_bots_server_bots():
    import pandas as pd
    db_from = Postgresql('botlorien')
    db_to = Postgresql('botlorienb')

    # Select the data from old database
    script = "select * from bots"
    data = db_from.execute_script(script)
    columns = db_from.select_columns('bots', False).replace('"', '').split(' ,')
    df = pd.DataFrame(data)
    df.columns = columns

    # Select data from new database
    script = "select * from bots"
    data = db_to.execute_script(script)
    columns2 = db_to.select_columns('bots', False).replace('"', '').split(' ,')
    df2 = pd.DataFrame(data)
    df2.columns = columns2

    # select the ids to ignore
    ignore_index = [i for i, id_ in enumerate(df[df.columns[0]]) if id_ not in df2[df2.columns[0]].tolist()]

    # delete ids to ignore from dataframe's old database
    table = df.loc[ignore_index].reset_index(drop=True)

    print(table)

    # insert the different data into the new database
    db_to.to_postgresql(table, 'bots', id_column=False)


def migrate_data_bots_server_tasks():
    import pandas as pd
    db_from = Postgresql('botlorien')
    db_to = Postgresql('botlorienb')

    # Select the data from old database
    script = "select * from tasks"
    data = db_from.execute_script(script)
    columns = db_from.select_columns('tasks', False).replace('"', '').split(' ,')
    df = pd.DataFrame(data)
    df.columns = columns

    # Select data from new database
    script = "select * from tasks"
    data = db_to.execute_script(script)
    columns2 = db_to.select_columns('tasks', False).replace('"', '').split(' ,')
    df2 = pd.DataFrame(data)
    df2.columns = columns2

    # select the ids to ignore
    ignore_index = [i for i, id_ in enumerate(df[df.columns[0]]) if id_ not in df2[df2.columns[0]].tolist()]

    # delete ids to ignore from dataframe's old database
    table = df.loc[ignore_index].reset_index(drop=True)

    print(table)

    # insert the different data into the new database
    db_to.to_postgresql(table, 'tasks', id_column=False)


def migrate_data_bots_server_tasks_results():
    import pandas as pd
    db_from = Postgresql('botlorien')
    db_to = Postgresql('botlorienb')

    # Select the data from old database
    script = "select * from tasks_results"
    data = db_from.execute_script(script)
    columns = db_from.select_columns('tasks_results', False).replace('"', '').split(' ,')
    df = pd.DataFrame(data)
    df.columns = columns

    # Select data from new database
    script = "select * from tasks_results"
    data = db_to.execute_script(script)
    columns2 = db_to.select_columns('tasks_results', False).replace('"', '').split(' ,')
    df2 = pd.DataFrame(data)
    df2.columns = columns2

    # select the ids to ignore
    ignore_index = [i for i, id_ in enumerate(df[df.columns[0]]) if id_ not in df2[df2.columns[0]].tolist()]

    # delete ids to ignore from dataframe's old database
    table = df.loc[ignore_index].reset_index(drop=True)

    # adjust some columns
    table['task_end_time'] = ['NULL' if str(val) == 'NaT' else val for val in table['task_end_time']]
    table['duration_task'] = ['NULL' if str(val) == 'None' else val for val in table['duration_task']]

    print(table)

    # insert the different data into the new database
    db_to.to_postgresql(table, 'tasks_results', id_column=False)

def teste_path():
    print(os.path.realpath(__file__))
    print(os.getcwd())

if __name__ == '__main__':
    print(os.path.dirname(os.path.realpath(__file__)))