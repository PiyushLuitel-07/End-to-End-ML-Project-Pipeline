import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_message

    

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    
'''

This code defines a custom exception class called CustomException and a function error_message_detail to generate detailed error messages.

Here's a breakdown of what each part of the code does:

import sys: This imports the sys module, which provides access to some variables used or maintained by the Python interpreter and functions that interact strongly with the interpreter. In this code, it's used for handling system-level operations.
from src.logger import logging: This imports the logging module from a package called src.logger. This suggests that there's a separate file or package named logger located in a directory named src from which the logging module is imported. This could be used for logging purposes.
def error_message_detail(error, error_detail: sys): This function takes two arguments, error and error_detail, and returns a detailed error message.
error is the exception that occurred.
error_detail is expected to be a sys module object (likely a sys.exc_info() result), which contains information about the exception.
Inside the function, it retrieves information such as the filename, line number, and the error message itself using the exc_info() method of the sys module. Then it formats these details into a string and returns it.
class CustomException(Exception): This defines a custom exception class named CustomException, which inherits from Python's built-in Exception class. This means CustomException can be used like any other exception type in Python.
__init__ method: This is the constructor method for CustomException. It takes two arguments, error_message and error_detail, where error_message is a string representing the error message, and error_detail is expected to be a sys module object.
Inside the constructor, it calls the parent class's (Exception) constructor using super().__init__(error_message), passing the error message to it. Then it assigns the detailed error message generated by error_message_detail to self.error_message.
__str__ method: This method returns a string representation of the exception. In this case, it returns the detailed error message stored in self.error_message.
Overall, this code provides a way to create custom exceptions with detailed error messages, including information like the filename, line number, and the error message itself. It's a useful pattern for debugging and error handling.
'''

if __name__=="__main__":
    try:
        a=1
        b='bv'
        c=a+b
    except Exception as e:
        logging.info("Different datatype cannot be added")
        raise CustomException(e,sys)