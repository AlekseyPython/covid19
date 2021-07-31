import re
import datetime
import numpy as np
from pandas import NA
import functools
from fuzzywuzzy import process
from Entires import SourceColumns as SC, ConvertedColumns as CC
from Entires.GeoCoordinatesFinder import GeoFinder
import Settings

    
def get_converter(column):
    if column in  [CC.Source, CC.Region, CC.StatusDecisionOfHospitalization, CC.StatusDecisionOfObservation, 
                   CC.TestInformation, CC.TypeOfPneumonia, CC.NameOfHospital]:
        return functools.partial(_get_category_without_replacement, column_name=column.get_name())
        
    elif column == CC.Country:
        return _get_country
        
    elif column == CC.Age:
        return _get_age
    
    elif column == CC.Death:
        return _get_death
    
    elif column == CC.Sex:
        return _get_sex
    
    elif column == CC.DecisionOfAmbulance:
        return _get_decision_of_ambulance
    
    elif column == CC.ZodiacSign:
        return _get_zodiac_sign
    
    elif column == CC.PrimaryElement:
        return _get_primary_element
    
    elif column in [CC.PeriodOfObservation, CC.PeriodOfQuarantine, CC.PeriodOfAftercare]:
        column_name = column.get_name().replace('PeriodOf', '')
        start_name='DateAdmissionTo' + column_name
        finish_name='DateDepartureFrom' + column_name
        return functools.partial(_get_period, start_name=start_name, finish_name=finish_name)
    
    elif column == CC.PeriodFromHospitalToAftercare:
        return _get_period_from_hospital_to_aftercare
        
    elif column == CC.PeriodFromAmbulanceToHospital:
        return _get_period_from_ambulance_to_hospital
                    
    elif column == CC.PeriodOfHospitalization:
        return _get_period_hospitalization
    
    elif column == CC.DateAdmissionToHospital:
        return _get_date_admission_to_hospital
    
    elif column == CC.DateDepartureFromHospital:
        return _get_date_departure_from_hospital
    
    elif column == CC.PeriodDiseaseForHospitalization:
        return _get_period_disease_for_hospitalization
    
    elif column in [CC.DateCreating, CC.DateAnalysis, CC.Birthday]:
        return functools.partial(_get_date, column_name=column.get_name())
    
    elif column == CC.Birthmonth:
        return _get_birthmonth
    
    elif column in [CC.WeekDayArrivalAmbulance, CC.WeekDayAdmissionToHospital, CC.WeekDayDepartureFromHospital]:
        sc_name = column.get_name().replace('WeekDay', 'Date')
        return functools.partial(_get_week_day, column_name=sc_name)
    
    elif column == CC.SeverityOfDisease:
        return _get_severity_of_disease
    
    elif column in CC.all_result:
        return functools.partial(_get_result, column_name=column.get_name())
    
    elif column == CC.GroupOfRisk:
        return _get_group_of_risk
    
    elif column in [CC.DIC, CC.MV, CC.ECMO, CC.DidNotTravel, CC.NotFoundAtHome, CC.AntiviralTreatment]:
        return functools.partial(_convert_yes_no, column_name=column.get_name())
    
    elif column in [CC.ImmunosuppressantsDrugs, CC.TreatmentHivInfectionDrugs, CC.AntiviralDrugs]:
        return functools.partial(_get_pharmacological_group, column_name=column.get_name())
        
    elif column == CC.TransferredToHospitalFromAnotherHospital:
        return _get_transferred_to_hospital_from_another_hospital
    
    elif column == CC.TransferredToHospitalFromQuarantine:
        return _get_transferred_to_hospital_from_quarantine
        
    elif column in [CC.SaturationLevel]:
        return functools.partial(_get_int8, column_name=column.get_name())
    
    elif column == CC.PhoneOperator:
        return _get_phone_operator
    
    elif column in [CC.Longitude, CC.Latitude]:
        geo_finder = GeoFinder(silent=True)
        priority_addresses = ['AddressArrival', 'AddressOfResidence']
        priority_addresses.extend(map(SC._Column.get_name, SC.all_comment))
        return functools.partial(_get_coordinate, geo_finder=geo_finder, priority_addresses=priority_addresses, coordinate_name=column.get_name())
    
    else:
        raise RuntimeError('An undefined column is used for the calculated column')

def _get_category_without_replacement(row, column_name):
    value = row[column_name]
    if value:
        return value
    return NA

def _get_country(row):
    value = row['Country']
    if not value:
        return NA
    
    value = value.upper()
    if value in ['РОССИЯ', 'УКРАИНА', 'АРМЕНИЯ', 'АЗЕРБАЙДЖАН', 'КАЗАХСТАН', 'ТАДЖИКИСТАН', 'КИРГИЗИЯ', 'ГРУЗИЯ', 'БЕЛАРУСЬ', 'АБХАЗИЯ', 'УЗБЕКИСТАН', 'МОЛДОВА, РЕСПУБЛИКА']:
        return NA
    return value
    
def _get_zodiac_sign(row):
    birthday = row['Birthday']
    if not birthday:
        return NA
    
    day = int(birthday[:2])
    month = int(birthday[3:5])
    if day==1 and month==1:
        #culling 1 january
        return NA
    
    if (month==12 and day>=23) or (month==1 and day<=20):
        index = 0
    elif (month==1 and day>=21) or (month==2 and day<=19):
        index = 1
    elif (month==2 and day>=20) or (month==3 and day<=20):
        index = 2
    elif (month==3 and day>=21) or (month==4 and day<=20):
        index = 3
    elif (month==4 and day>=21) or (month==5 and day<=21):
        index = 4
    elif (month==5 and day>=22) or (month==6 and day<=21):
        index = 5
    elif (month==6 and day>=22) or (month==7 and day<=22):
        index = 6
    elif (month==7 and day>=23) or (month==8 and day<=21): 
        index = 7
    elif (month==8 and day>=22) or (month==9 and day<=23): 
        index = 8
    elif (month==9 and day>=24) or (month==10 and day<=23):
        index = 9
    elif (month==10 and day>=24) or (month==11 and day<=22): 
        index = 10
    elif (month==11 and day>=23) or (month==12 and day<=22):
        index = 11
    return Settings.VALUES_ZODIAC_SIGNS[index]

def _get_primary_element(row):
    zodiac_sign = _get_zodiac_sign(row)
    if zodiac_sign is NA:
        return NA
    
    if zodiac_sign in ['Aries', 'Leo', 'Sagittarius']:
        return 'Fire'
    
    elif zodiac_sign in ['Gemini', 'Libra', 'Aquarius']:
        return 'Air'
    
    elif zodiac_sign in ['Taurus', 'Virgo', 'Capricorn']:
        return 'Ground'
    
    elif zodiac_sign in ['Cancer', 'Scorpio', 'Pisces']:
        return 'Water'
    
    else:
        raise RuntimeError('It is impossible to determine the element for this zodiac sign: ' + zodiac_sign)
    
def _get_source(row):
    return row['Source']

def _get_age(row):
    date_disease = _get_date(row, 'DateCreating')
    if date_disease == Settings.EMPTY_DATE:
        date_disease = _get_date(row, 'DateAnalysis')
        if date_disease == Settings.EMPTY_DATE:
            return Settings.EMPTY_INT
        
    birthday = _get_date(row, 'Birthday')
    if birthday == Settings.EMPTY_DATE:
        return Settings.EMPTY_INT
    
    period = (date_disease - birthday)/np.timedelta64(1, 'D')
    if period < 0:
        return Settings.EMPTY_INT
    
    period /= 365.25
    if period > 127:
        return Settings.EMPTY_INT
    
    return np.int8(period)

def _get_period(row, start_name, finish_name):
    start = _get_date(row, start_name)
    if start==Settings.EMPTY_DATE or start<Settings.FIRST_DATE or start>Settings.LAST_DATE:
        return Settings.EMPTY_INT
    
    finish = _get_date(row, finish_name)
    if finish==Settings.EMPTY_DATE or finish<Settings.FIRST_DATE or finish>Settings.LAST_DATE:
        return Settings.EMPTY_INT
    
    period = (finish - start)/np.timedelta64(1, 'D')
    if period < 0:
        return Settings.EMPTY_INT
    
    return np.int8(period)

def _get_period_hospitalization(row):
    start = _get_date(row, 'DateAdmissionToHospital')
    if start == Settings.EMPTY_DATE:
        start = _get_date(row, 'DateArrivalAmbulance')
        if start == Settings.EMPTY_DATE:
            return Settings.EMPTY_INT
        else:
            start += Settings.PERIOD_FROM_AMBULANCE_TO_HOSPITALIZATION
    
    if start<Settings.FIRST_DATE or start>Settings.LAST_DATE:
        return Settings.EMPTY_INT
    
    finish = _get_date(row, 'DateDepartureFromHospital')
    if finish == Settings.EMPTY_DATE:
        finish = _get_date(row, 'DateAdmissionToAftercare')
        if finish == Settings.EMPTY_DATE:
            return Settings.EMPTY_INT
        else:
            finish -= Settings.PERIOD_FROM_HOSPITALIZATION_TO_AFTERCARE
    
    if finish<Settings.FIRST_DATE or finish>Settings.LAST_DATE:
        return Settings.EMPTY_INT
    
    period = (finish - start)/np.timedelta64(1, 'D')
    if period < 0:
        return Settings.EMPTY_INT
    
    return np.int8(period)

def _get_date_admission_to_hospital(row):
    start = _get_date(row, 'DateAdmissionToHospital')
    if start == Settings.EMPTY_DATE:
        if not _exist_hospitalization(row):
            return Settings.EMPTY_DATE
        
        start = _get_date(row, 'DateArrivalAmbulance')
        if start == Settings.EMPTY_DATE:
            return Settings.EMPTY_DATE
        else:
            start += Settings.PERIOD_FROM_AMBULANCE_TO_HOSPITALIZATION
                
    if start<Settings.FIRST_DATE or start>Settings.LAST_DATE:
        return Settings.EMPTY_DATE
    return start
    
def _get_date_departure_from_hospital(row):
    finish = _get_date(row, 'DateDepartureFromHospital')
    if finish == Settings.EMPTY_DATE:
        if not _exist_hospitalization(row):
            return Settings.EMPTY_DATE
        
        finish = _get_date(row, 'DateAdmissionToAftercare')
        if finish == Settings.EMPTY_DATE:
            return Settings.EMPTY_DATE
        else:
            finish -= Settings.PERIOD_FROM_HOSPITALIZATION_TO_AFTERCARE
                
    if finish<Settings.FIRST_DATE or finish>Settings.LAST_DATE:
        return Settings.EMPTY_DATE
    return finish

def _get_period_from_hospital_to_aftercare(row):
    if not _exist_hospitalization(row):
        return Settings.EMPTY_INT
    return _get_period(row, 'DateDepartureFromHospital', 'DateAdmissionToAftercare')

def _get_period_from_ambulance_to_hospital(row):
    decision_of_ambulance = _get_decision_of_ambulance(row)
    if decision_of_ambulance is NA or decision_of_ambulance != 'Стационар':
        return Settings.EMPTY_INT
    return _get_period(row, 'DateArrivalAmbulance', 'DateAdmissionToHospital')
    
def _exist_hospitalization(row):
    decision_of_ambulance = _get_decision_of_ambulance(row)
    if decision_of_ambulance is NA or decision_of_ambulance != 'Стационар':
        transferred_to_hospital_from_quarantine = _get_transferred_to_hospital_from_quarantine(row)
        if transferred_to_hospital_from_quarantine is NA or not transferred_to_hospital_from_quarantine:
            return False
    return True
    
def _get_period_disease_for_hospitalization(row):
    start = _get_date(row, 'DateCreating')
    if start==Settings.EMPTY_DATE or start<Settings.FIRST_DATE or start>Settings.LAST_DATE:
        return Settings.EMPTY_INT
            
    finish = _get_date_departure_from_hospital(row)
    if finish==Settings.EMPTY_DATE or finish<Settings.FIRST_DATE or finish>Settings.LAST_DATE:
        return Settings.EMPTY_INT
        
    period = (finish - start)/np.timedelta64(1, 'D')
    if period < 0:
        return Settings.EMPTY_INT
    return np.int8(period)
    
def _get_date(row, column_name):
    current_date = row[column_name]
    if not current_date:
        return Settings.EMPTY_DATE
    
    day = current_date[:2]
    month = current_date[3:5]
    if day=='01' and month=='01':
        #culling 1 january
        return Settings.EMPTY_DATE
    
    year = current_date[6:]
    year_i = int(year)
    if year_i < 1900 or year_i > 2020:
        return Settings.EMPTY_DATE
    
    return np.datetime64(year + '-' + month + '-' + day, 'D')

def _get_birthmonth(row):
    birthday = row['Birthday']
    if not birthday:
        return -1
    
    day = int(birthday[:2])
    month = int(birthday[3:5])
    if day==1 and month==1:
        #culling 1 january
        return -1
    
    return month

def _get_week_day(row, column_name):
    current_date = row[column_name]
    if not current_date:
        return -1
    
    day = int(current_date[:2])
    month = int(current_date[3:5])
    if day==1 and month==1:
        #culling 1 january
        return -1
    
    year = int(current_date[6:])
    if year<1900 or year>2020:
        return -1
     
    current_date = datetime.date(year, month, day)
    weekday = current_date.weekday() + 1 #1, 2,..., 7
    return weekday
    
def _get_death(row):
    #bool stored as int8
    for column in SC.all_result:
        value = row[column.get_name()].upper()
        if value.find('УМЕР') > -1:
            return 1
        
    #free-form text fields
    words_of_death = ['УМЕР', 'СКОНЧАЛ', 'СМЕРТ']
    for column in SC.all_comment:
        value = row[column.get_name()].upper()
        for word in words_of_death:
            if value.find(word) > -1:
                return 1
    return 0

def _get_sex(row):
    current_value = row['MiddleNameEnding']
    if current_value:
        return Settings.REPLACEMENTS_SEX.get(current_value, NA)
    return NA
    
def _get_decision_of_ambulance(row):
    current_value = row['DecisionOfAmbulance']
    if current_value:
        return Settings.REPLACEMENTS_DECISION_OF_AMBULANCE.get(current_value, current_value)
    
    #after hospitalizataion and obervation patients are taken to aftercare
    for name in SC.all_names_quarantine:
        if row[name]:
            return 'Домашний карантин'
        
    if row['ResultOfObservation'] in ['Переведен в стационар']:
        return 'Обсервация'
    
    #transfer from a hospital to an obervation occurs more often than in the opposite direction
    for name in SC.all_names_hospitalizataion:
        if name in ['DIC', 'MV']:
            if row[name] and row[name] != 'Нет':
                return 'Стационар'
        else:
            if row[name]:
                return 'Стационар'
            
    for name in SC.all_names_obervation:
        if row[name]:
            return 'Обсервация'
    return NA

def _get_severity_of_disease(row):
    columns = []
    columns.append('SeverityOfDiseaseAmbulance')
    columns.append('SeverityOfDiseaseToHospital')
    columns.append('SeverityOfDiseaseInHospital')
    columns.append('SeverityOfDiseaseQuarantine')
    columns.append('SeverityOfDiseaseCT')
    return _get_max_severity_of_disease(columns, row)

def _get_max_severity_of_disease(columns, row):
    indexes = []
    for column in columns:
        value = row[column]
        if value:
            value = Settings.REPLACEMENTS_SEVERITY_OF_DISEASE.get(value, value)
            index = Settings.VALUES_SEVERITY_OF_DISEASE.index(value)
            indexes.append(index)
    
    if indexes:
        max_index = max(indexes)
        return Settings.VALUES_SEVERITY_OF_DISEASE[max_index]
    else:
        return NA
    
def _get_result(row, column_name):
    current_value = row[column_name]
    if current_value:
        return Settings.REPLACEMENTS_RESULT.get(current_value, current_value)
    return NA

def _get_group_of_risk(row):
    columns = []
    columns.append('GroupOfRiskHospitalization')
    columns.append('GroupOfRiskObservation')
    columns.append('GroupOfRiskQuarantine')
    columns.append('GroupOfRiskCT')
    return _get_group_of_risk_for_columns(columns, row)

def _get_group_of_risk_for_columns(columns, row):
    for column in columns:
        value = row[column]
        if value:
            value = Settings.REPLACEMENTS_GROUP_OF_RISK.get(value, value)
            if value is not NA:
                return value
    return NA

def _convert_yes_no(row, column_name):
    #bool stored as int8
    value = row[column_name].upper()
    if value == 'ДА':
        return 1
    elif value == 'НЕТ':
        return 0
    else:
        return Settings.EMPTY_BOOL

def _get_pharmacological_group(row, column_name):
    if column_name == 'ImmunosuppressantsDrugs':
        dictionary = Settings.IMMUNOSUPPRESSANT_DRUGS
        
    elif column_name == 'TreatmentHivInfectionDrugs':
        dictionary = Settings.TREATMENT_HIV_INFECTION_DRUGS
        
    elif column_name == 'AntiviralDrugs':
        dictionary = Settings.ANTIVIRAL_DRUGS
        
    else:
        raise RuntimeError('No pharmacological group defined for your data column!')
    
    table_of_simbols = {}
    table_of_simbols[ord(',')] = ord(' ')
    table_of_simbols[ord('.')] = ord(' ')
    table_of_simbols[ord(';')] = ord(' ')
    
    value = row['ListOfMedicines'].lower()
    value = value.translate(table_of_simbols)
    words_of_string = value.split()
    
    for word in words_of_string:
        if len(word) < 7:
            continue
        
        _, distance = process.extractOne(word, dictionary)
        if distance > 75:
            return 1
    return Settings.EMPTY_BOOL
                
        
def _get_transferred_to_hospital_from_another_hospital(row):
    #bool stored as int8
    if row['TransferDate'] or row['InitialHospital']:
        return 1
    else:
        decision = _get_decision_of_ambulance(row)
        if decision is NA:
            return Settings.EMPTY_BOOL
        elif decision == 'Стационар':
            return 0
        else:
            return Settings.EMPTY_BOOL
        
def _get_transferred_to_hospital_from_quarantine(row):
    decision_of_ambulance = _get_decision_of_ambulance(row)
    if decision_of_ambulance is NA or decision_of_ambulance != 'Домашний карантин':
        return False
    
    for name in SC.all_names_hospitalizataion:
        if name in ['DIC', 'MV']:
            if row[name] != 'Нет':
                return True
        else:
            if row[name]:
                return True
    return False
               
def _get_int8(row, column_name):
    value = row[column_name]
    if value:
        return np.int8(value)
    return Settings.EMPTY_INT

def _get_phone_operator(row):
    result = _get_phone_operator_from_column(row['AddressOfResidence'])
    if result is not None:
        return result
    
    for column in SC.all_comment:
        column_name = column.get_name()
        result = _get_phone_operator_from_column(row[column_name])
        if result is not None:
            return result
    return ''
        
def _get_phone_operator_from_column(column_value):
    if not column_value:
        return None
    
    #some numbers in phone number are replaced with stars! First number always 4 or 9
    pattern = '[78\*]{1} [-\s]* [\(]? ([49\*]{1} [\d\*]{2}) [\)]? [-\s]* [\d\*]{3} [-\s]* [\d\*]{2} [-\s]* [\d\*]{2}'
    pattern = pattern.replace(' ', '')
    result = re.search(pattern, column_value)
    
    if result is None:
        #without leading 7 or 8
        pattern = '[\(]? ([49\*]{1} [\d\*]{2}) [\)]? [-\s]* [\d\*]{3} [-\s]* [\d\*]{2} [-\s]* [\d\*]{2}'
        pattern = pattern.replace(' ', '')
        result = re.search(pattern, column_value)
        
        if result is None:
            return None
        
    phone_code = result.group(1)
    if phone_code[0] == '4':
        return 'МГТС'
    
    phone_code = phone_code[1:]
    star_position = phone_code.find('*')
    if star_position == -1:
        return Settings.PHONE_OPERATOR_CODES.get(int(phone_code), None)
    
    second_number = phone_code[0]    
    if second_number == '*':
        return None
    else:
        return Settings.PHONE_OPERATOR_CODES_SECOND_NUMBER.get(int(second_number), None)
    
def _get_coordinate(row, geo_finder, priority_addresses, coordinate_name):
    if row['Region'] in ['Московская область', 'Другие регионы']:
        return None
    
    fields = [row[field_name] for field_name in priority_addresses if row[field_name]]
    coordinates = geo_finder.get_coordinates(fields)
    if coordinates is None:
        return None
    
    if coordinate_name == 'Latitude':
        return coordinates[1]
    else:
        return coordinates[0]
    
    
    


