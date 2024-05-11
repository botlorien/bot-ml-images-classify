from scripts.database import BotTable
import logging, time
from functools import wraps
TIMEOUT = 20


class BotApp:

    def __init__(self):
        """
        Initialize the BotApp object.

        """
        self.bot = BotTable()
        self.bot_name = "My_Bot_Name"  # Provide bot name
        self.bot_description = "This bot does XYZ tasks."  # Provide bot description
        self.bot_version = "1.0.0"  # Provide bot version MAJOR.MINOR.PATCH
        self.bot_department = ''
        self.user_name = ""  # Provide user name
        self.error = None
        self.result = None
        self.status = None

    def set_bot_name(self, bot_name):
        """
        Set the name of the bot.

        :param bot_name: The name of the bot.
        :type bot_name: str
        """
        self.bot_name = bot_name

    def set_bot_description(self, description):
        """
        Set the description of the bot.

        :param description: The description of the bot.
        :type description: str
        """
        self.bot_description = description

    def set_bot_version(self, version):
        """
        Set the version of the bot.

        :param version: The version of the bot.
        :type version: str
        """
        self.bot_version = version

    def set_bot_department(self, department):
        """
        Set the department of the bot.

        :param department: The department of the bot.
        :type department: str
        """
        self.bot_department = department

    def set_user(self, user):
        """
        Set the user of the bot.

        :param user: The user of the bot.
        :type user: str
        """
        self.user_name = user

    def init_bot(self):
        """
        Initialize the bot by registering the bot name and description.

        """
        self.bot.register_bot(self.bot_name, self.bot_description)

    def time_out(void=None,raise_exception: bool = True):
        """
        Decorator that sets a timeout for a function.

        :param time_out: The timeout duration in seconds.
        :type time_out: int
        :param raise_exception: Flag to raise an exception if the timeout is exceeded.
        :type raise_exception: bool
        """

        def wrapper(func):
            def inner_wrapper(*args, **kwargs):
                contador_time_out = 0
                ret = False
                error = None
                while contador_time_out < TIMEOUT:
                    try:
                        ret = func(*args, **kwargs)
                        break
                    except Exception as e:
                        error = e
                        logging.exception(error)
                        time.sleep(1)
                    contador_time_out += 1

                    if contador_time_out >= TIMEOUT and raise_exception:
                        raise error
                return ret

            return inner_wrapper

        return wrapper

    @time_out()
    def set_time_out(self, task_function, *args, **kwargs):
        """
        Set a timeout for executing a task function.

        :param task_function: The task function to execute.
        :type task_function: function
        :param args: Arguments for the task function.
        :param kwargs: Keyword arguments for the task function.
        :return: The result of the task function.
        """
        return task_function(*args, **kwargs)

    def _run_task(self, task_function, *args, **kwargs):
        """
        Run a task function with error handling.

        :param task_function: The task function to run.
        :type task_function: function
        :param args: Arguments for the task function.
        :param kwargs: Keyword arguments for the task function.
        """
        try:
            # ensuring that this values is set with th corrent values
            self.result = None
            self.status = None
            self.error = None

            self.result = self.set_time_out(task_function, *args, **kwargs)
            self.status = 'success'
        except Exception as e:
            self.error = e
            self.status = 'failed'

    def set_task(self, task_name, task_description):
        """
        Register a task with its name and description.

        :param task_name: The name of the task.
        :type task_name: str
        :param task_description: The description of the task.
        :type task_description: str
        """
        self.bot.register_task(task_name, task_description)

    def execute_bot_task(self, task_function, *args, **kwargs):
        """
        Execute a bot task.

        :param task_function: The task function to execute.
        :type task_function: function
        :param args: Arguments for the task function.
        :param kwargs: Keyword arguments for the task function.
        :return: The result of the task function.
        """
        global TIMEOUT
        TIMEOUT = TIMEOUT if 'timeout' not in kwargs else kwargs.pop('timeout')
        self.bot.create_new_instance_from_task(self.bot_version, self.user_name)
        self.bot.open_task()
        self._run_task(task_function, *args, **kwargs)
        self._end_bot_task()
        return self.result

    def _end_bot_task(self):
        """
        End the bot task and handle errors.

        :param kwargs: Keyword arguments.
        """
        result = str(self.result).replace("'", "`").replace('"', "`") if self.result is not None else ''
        erro = str(self.error).replace("'", "`").replace('"', "`") if self.error is not None else ''
        self.bot.end_task(self.status, result, erro)
        if self.error is not None:
            raise self.error

    def set_bot(self,bot_name,bot_description,bot_version,bot_department):
        self.set_bot_name(bot_name)
        self.set_bot_description(
            bot_description)
        self.set_bot_version(bot_version)
        self.set_bot_department(bot_department)
        self.init_bot()

    def task(self,func=None):
        @wraps(func)
        def wrapper(*args,**kwargs):
            self.set_task(func.__name__, func.__doc__)
            return self.execute_bot_task(func)
        return wrapper


if __name__ == "__main__":
    pass
