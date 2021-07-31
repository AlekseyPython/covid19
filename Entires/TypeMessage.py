from enum import Enum

   
class TypeMessage(Enum):
    information = 'Information'
    warning = 'Warning'
    critical = 'Critical'
    question = 'Question'