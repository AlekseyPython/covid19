import os
from pandas import  NA
from numpy import datetime64

EMPTY_DATE = datetime64('1800-01-01', 'D')
EMPTY_INT = -1
EMPTY_BOOL = -1

CENTER_OF_MOSCOW = (37.6194, 55.7519)

FIRST_DATE = datetime64('2020-02-01', 'D')
LAST_DATE = datetime64('2020-05-04', 'D')
LAST_FULL_MONTH = datetime64('2020-04', 'M')

PERIOD_FROM_HOSPITALIZATION_TO_AFTERCARE = 1
PERIOD_FROM_AMBULANCE_TO_HOSPITALIZATION = 0

MINIMUM_QUANTITY_STATIONS_FOR_CALCULATING_AIR_POLLUTION = 10

console_mode = False
debuge_mode = False

PATH_COVID_PATIENTS = '/home/ivan/Documents/MoscowBase/CovidPatients.csv'
CONVERTED_DATA = '/home/ivan/Documents/MoscowBase/converted_data.h5'
FOLDER_FOR_TEMPORARY_FILES = '/home/ivan/Documents/MoscowBase/TEMPORARY_FILES'
FOLDER_MOSCOW_MAPS = '/home/ivan/Documents/MoscowBase/MOSCOW_MAPS'
FOLDER_WITH_IMAGES = '/home/ivan/Documents/MoscowBase/IMAGES'

PATH_BASE_COORDINATES_OF_ADDRESSES = '/home/ivan/Documents/MoscowBase/Osmand/BASE_COORDINATES_OF_ADDRESSES.csv'
PATH_BASE_TYPE_BUILDING_AND_COORDINATES = '/home/ivan/Documents/MoscowBase/Osmand/BASE_TYPE_BUILDING_AND_COORDINATES.csv'

PATH_BASE_AIR_POLLUTIONS = '/home/ivan/Documents/MoscowBase/Air pollution/values.csv'
PATH_BASE_AIR_STATIONS = '/home/ivan/Documents/MoscowBase/Air pollution/stations.csv'
PATH_BASE_PARAMETERS_POLLUTIONS = '/home/ivan/Documents/MoscowBase/Air pollution/parameters.csv'

REPLACEMENTS_SEVERITY_OF_DISEASE = {}
REPLACEMENTS_SEVERITY_OF_DISEASE['Легкая степень тяжести'] = 'Легкое течение'
REPLACEMENTS_SEVERITY_OF_DISEASE['Средняя степень тяжести'] = 'Средней тяжести'
REPLACEMENTS_SEVERITY_OF_DISEASE['Тяжелая степень тяжести'] = 'Тяжелое течение'

VALUES_SEVERITY_OF_DISEASE = ['Без симптомов']
VALUES_SEVERITY_OF_DISEASE.extend(REPLACEMENTS_SEVERITY_OF_DISEASE.values())
VALUES_SEVERITY_OF_DISEASE_ODERED = True


REPLACEMENTS_RESULT = {}
REPLACEMENTS_RESULT['лечится'] = 'Лечится'
REPLACEMENTS_RESULT['Отправлен на долечивание на домашний карантин'] = 'Домашний карантин'
REPLACEMENTS_RESULT['Госпитализация'] = 'Cтационар'
REPLACEMENTS_RESULT['Госпитализирован в стационар'] = 'Cтационар'
REPLACEMENTS_RESULT['Переведен в стационар'] = 'Cтационар'
REPLACEMENTS_RESULT['Госпитализирован в обсерватор'] = 'Обсервация'
REPLACEMENTS_RESULT['Переведен в обсерватор'] = 'Обсервация'
REPLACEMENTS_RESULT['Умер (прочие причины)'] = 'Умер'
REPLACEMENTS_RESULT['Умер (причина уточняется)'] = 'Умер'
REPLACEMENTS_RESULT['Умер (COVID)'] = 'Умер'
REPLACEMENTS_RESULT['Пациент по данному адресу отсутствует'] = NA

VALUES_RESULT = ['Выздоровел']
VALUES_RESULT.extend(set(REPLACEMENTS_RESULT.values()) - set([NA]))
VALUES_RESULT_ODERED = False


REPLACEMENTS_DECISION_OF_AMBULANCE = {}
REPLACEMENTS_DECISION_OF_AMBULANCE['Госпитализация'] = 'Стационар'
REPLACEMENTS_DECISION_OF_AMBULANCE['тяжелое течение'] = NA
REPLACEMENTS_DECISION_OF_AMBULANCE['Решение не требуется'] = NA

VALUES_DECISION_OF_AMBULANCE = ['Домашний карантин', 'Обсервация', 'Направление в КТ-Центр']
VALUES_DECISION_OF_AMBULANCE.extend(set(REPLACEMENTS_DECISION_OF_AMBULANCE.values()) - set([NA]))
VALUES_DECISION_OF_AMBULANCE_ODERED = False


REPLACEMENTS_SEX = {}
REPLACEMENTS_SEX['вич'] = 'Men'
REPLACEMENTS_SEX['вна'] = 'Women'

VALUES_SEX = []
VALUES_SEX.extend(REPLACEMENTS_SEX.values())
VALUES_SEX_ODERED = False

REPLACEMENTS_GROUP_OF_RISK = {}
REPLACEMENTS_GROUP_OF_RISK['Не относится к группе риска'] = NA
REPLACEMENTS_GROUP_OF_RISK['Старше 65+'] = NA

VALUES_GROUP_OF_RISK = ['Хронический больной', 'Относится к группе риска', 'Беременность']
VALUES_GROUP_OF_RISK_ODERED = False

VALUES_ZODIAC_SIGNS = ['Capricorn', 'Aquarius', 'Pisces', 'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius']
VALUES_ZODIAC_SIGNS_ODERED = False

VALUES_PRIMARY_ELEMENT = ['Fire', 'Air', 'Ground', 'Water']
VALUES_PRIMARY_ELEMENT_ODERED = False

REPLACEMENTS_LIST_OF_MEDICINES = {}
REPLACEMENTS_LIST_OF_MEDICINES['кислоты'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['сульфат'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['получает'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['препараты'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['лекарственные'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['продолжительность'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['капельно'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['кратность'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['лечение'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['внутрь'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['лекарственных'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['назначено'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['перорально'] = ''
REPLACEMENTS_LIST_OF_MEDICINES['препаратов'] = ''

IMMUNOSUPPRESSANT_DRUGS = ['плаквенил', 'гидроксихлорохин']
TREATMENT_HIV_INFECTION_DRUGS = ['калетра', 'лопинавир', 'ритонавир']
ANTIVIRAL_DRUGS = ['ингавирин', 'имидазолилэтанамид', 'пентандиовой', 'арпефлю', 'осельтамивир', 'арбидол', 'умифеновир']

VALUES_TEST_INFORMATION = ['Результатов тестов нет', 'Есть один положительный', 'Все тесты отрицательные']
VALUES_TEST_INFORMATION_ODERED = False

VALUES_TYPES_OF_PNEUMONIA = ['Вирусная COVID', 'Вирусная иная', 'Бактериальная']
VALUES_TYPES_OF_PNEUMONIA_ODERED = False

VALUES_SOURCE = []
VALUES_SOURCE.append('КГУ')
VALUES_SOURCE.append('КТ')
VALUES_SOURCE.append('СТ+КГУ')
VALUES_SOURCE.append('СТ')
VALUES_SOURCE.append('КТ+КГУ')
VALUES_SOURCE.append('СТ+КТ')
VALUES_SOURCE_ODERED = False

VALUES_REGION = []
VALUES_REGION.append('Москва')
VALUES_REGION.append('Московская область')
VALUES_REGION.append('Другие регионы')
VALUES_REGION_ODERED = False


VALUES_STATUS_DECISION = []
VALUES_STATUS_DECISION.append('Добровольно')
VALUES_STATUS_DECISION.append('Принудительно')
VALUES_STATUS_DECISION_ODERED = False

ROOT_DIR = os.path.dirname(__file__)
def get_hospitals():
    with open(ROOT_DIR + '/Dictionaries/hospitals.txt', 'r') as text_file:
        lines = text_file.readlines()
    
    lines = list(map(str.strip, lines))
    return lines

VALUES_NAME_OF_HOSPITAL = get_hospitals()
VALUES_NAME_OF_HOSPITAL_ODERED = False

def get_countries():
    with open(ROOT_DIR + '/Dictionaries/countries.txt', 'r') as text_file:
        lines = text_file.readlines()
    lines = list(map(str.strip, lines))
    return lines

VALUES_COUNTRY = get_countries()
VALUES_COUNTRY_ODERED = False


PHONE_OPERATORS = ['МГТС', 'Билайн', 'МТС', 'Мегафон', 'TELE2']
PHONE_OPERATORS_ODERED = False

PHONE_OPERATOR_CODES = {}
for i in range(10):
    PHONE_OPERATOR_CODES[i] = 'Билайн'
    PHONE_OPERATOR_CODES[10 + i] = 'МТС'
    PHONE_OPERATOR_CODES[20 + i] = 'Мегафон'
    PHONE_OPERATOR_CODES[60 + i] = 'Билайн'


PHONE_OPERATOR_CODES[1] = 'TELE2'
PHONE_OPERATOR_CODES[30] = 'TELE2'
PHONE_OPERATOR_CODES[36] = 'Мегафон'
PHONE_OPERATOR_CODES[58] = 'МГТС'
PHONE_OPERATOR_CODES[77] = 'TELE2'
PHONE_OPERATOR_CODES[80] = 'Билайн'
PHONE_OPERATOR_CODES[83] = 'Билайн'
PHONE_OPERATOR_CODES[85] = 'МТС'
PHONE_OPERATOR_CODES[86] = 'МТС'
PHONE_OPERATOR_CODES[91] = 'TELE2'
PHONE_OPERATOR_CODES[95] = 'МГТС'
PHONE_OPERATOR_CODES[99] = 'МГТС'#999-Мегафон, but 999 value occurs half as often 

PHONE_OPERATOR_CODES_SECOND_NUMBER = {} 
PHONE_OPERATOR_CODES_SECOND_NUMBER[0] = 'Билайн'  
PHONE_OPERATOR_CODES_SECOND_NUMBER[1] = 'МТС'  
PHONE_OPERATOR_CODES_SECOND_NUMBER[2] = 'Мегафон'  
PHONE_OPERATOR_CODES_SECOND_NUMBER[3] = 'TELE2' #936 Мегафон, but it values 3 times less often
PHONE_OPERATOR_CODES_SECOND_NUMBER[5] = 'МГТС'  
PHONE_OPERATOR_CODES_SECOND_NUMBER[6] = 'Билайн'  
PHONE_OPERATOR_CODES_SECOND_NUMBER[7] = 'TELE2'  
PHONE_OPERATOR_CODES_SECOND_NUMBER[8] = 'МТС' #980 and 983- Билайн, but it values 32 times less often 
PHONE_OPERATOR_CODES_SECOND_NUMBER[9] = 'МГТС' #991-TELE2, but MGTS twice as often as TELE2

    

    

