from .SimilarColumns import SimilarColumns
import Settings


class _Column:
    def __init__(self, name, col_type):
        self.name = name
        self.col_type = col_type

    def get_name(self):
        return self.name
    
    def get_type(self):
        return self.col_type
    
    def get_predictable(self):
        return self.predictable
    
    def get_possible_values(self):
        if self.bool_type():
            return ['True', 'False', '<NA>']
            
        elif self.col_type == 'category':
            values, _ = self.get_categories()
            values.append('<NA>')
            return values
            
    def get_categories(self):
        if self == Source:
            return (Settings.VALUES_SOURCE.copy(), Settings.VALUES_SOURCE_ODERED)
        
        elif self == ZodiacSign:
            return (Settings.VALUES_ZODIAC_SIGNS.copy(), Settings.VALUES_ZODIAC_SIGNS_ODERED)

        elif self == PrimaryElement:
            return (Settings.VALUES_PRIMARY_ELEMENT.copy(), Settings.VALUES_PRIMARY_ELEMENT_ODERED)
        
        elif self == Region:
            return (Settings.VALUES_REGION.copy(), Settings.VALUES_REGION_ODERED)
        
        elif self == Sex:
            return (Settings.VALUES_SEX.copy(), Settings.VALUES_SEX_ODERED)
        
        elif self == DecisionOfAmbulance:
            return (Settings.VALUES_DECISION_OF_AMBULANCE.copy(), Settings.VALUES_DECISION_OF_AMBULANCE_ODERED)
        
        elif self in [StatusDecisionOfHospitalization, StatusDecisionOfObservation]:
            return (Settings.VALUES_STATUS_DECISION.copy(), Settings.VALUES_STATUS_DECISION_ODERED)
            
        elif self == NameOfHospital:
            return (Settings.VALUES_NAME_OF_HOSPITAL.copy(), Settings.VALUES_NAME_OF_HOSPITAL_ODERED)
        
        elif self == TestInformation:
            return (Settings.VALUES_TEST_INFORMATION.copy(), Settings.VALUES_TEST_INFORMATION_ODERED)
        
        elif self == TypeOfPneumonia:
            return (Settings.VALUES_TYPES_OF_PNEUMONIA.copy(), Settings.VALUES_TYPES_OF_PNEUMONIA_ODERED)
        
        elif self in all_result:
            return (Settings.VALUES_RESULT.copy(), Settings.VALUES_RESULT_ODERED)
            
        elif self == SeverityOfDisease:
            return (Settings.VALUES_SEVERITY_OF_DISEASE.copy(), Settings.VALUES_SEVERITY_OF_DISEASE_ODERED)
        
        elif self == GroupOfRisk:
            return (Settings.VALUES_GROUP_OF_RISK.copy(), Settings.VALUES_GROUP_OF_RISK_ODERED)
        
        elif self == Country:
            return (Settings.VALUES_COUNTRY.copy(), Settings.VALUES_COUNTRY_ODERED)
        
        elif self == Country:
            return (Settings.VALUES_COUNTRY.copy(), Settings.VALUES_COUNTRY_ODERED)
        
        elif self == PhoneOperator:
            return (Settings.PHONE_OPERATORS.copy(), Settings.PHONE_OPERATORS_ODERED)
        
        else:
            RuntimeError('A non-categorical data type passed to get categorical values!')
    
    def get_compare_operations(self):
        operations = []
        operations.append('=')
        operations.append('!=')
        
        if self.bool_type():
            exist_compare = False
        elif self.numeric_type() or self.date_type():
            exist_compare = True
        else:
            _, odered = self.get_categories()
            exist_compare = odered
                
        if exist_compare:
            operations.append('<')
            operations.append('<=')
            operations.append('>')
            operations.append('>=')
                
        return operations
            
    def numeric_type(self):
        if self.col_type.find('int')>=0 or self.col_type.find('float')>=0:
            return True
        return False
    
    def date_type(self):
        if self.col_type.find('datetime') >= 0:
            return True
        return False
    
    def bool_type(self):
        if self.col_type.find('bool') >= 0:
            return True
        return False

DidNotTravel =                  _Column('DidNotTravel', 'bool')
PhoneOperator =                 _Column('PhoneOperator', 'category')
Source =                        _Column('Source', 'category')
Death =                         _Column('Death', 'bool')
ZodiacSign =                    _Column('ZodiacSign', 'category')
PrimaryElement =                _Column('PrimaryElement', 'category')
Region =                        _Column('Region', 'category')
PeriodOfObservation =           _Column('PeriodOfObservation', 'int8') 
PeriodOfQuarantine =            _Column('PeriodOfQuarantine', 'int8')
PeriodOfHospitalization =       _Column('PeriodOfHospitalization', 'int8') 
PeriodOfAftercare =             _Column('PeriodOfAftercare', 'int8')
PeriodFromHospitalToAftercare = _Column('PeriodFromHospitalToAftercare', 'int8')
PeriodFromAmbulanceToHospital = _Column('PeriodFromAmbulanceToHospital', 'int8')
DateAdmissionToHospital =       _Column('DateAdmissionToHospital', 'datetime64[D]')
DateDepartureFromHospital =     _Column('DateDepartureFromHospital', 'datetime64[D]')
PeriodDiseaseForHospitalization=_Column('PeriodDiseaseForHospitalization', 'int8')
Birthmonth =                    _Column('Birthmonth', 'int8')
WeekDayArrivalAmbulance =       _Column('WeekDayArrivalAmbulance', 'int8')
WeekDayAdmissionToHospital =    _Column('WeekDayAdmissionToHospital', 'int8')
WeekDayDepartureFromHospital =  _Column('WeekDayDepartureFromHospital', 'int8')
Sex =                           _Column('Sex', 'category')
DateCreating =                  _Column('DateCreating', 'datetime64[D]')
Birthday =                      _Column('Birthday', 'datetime64[D]')
DateAnalysis =                  _Column('DateAnalysis', 'datetime64[D]')
Age =                           _Column('Age', 'int8')
NotFoundAtHome =                _Column('NotFoundAtHome', 'bool')
DecisionOfAmbulance =           _Column('DecisionOfAmbulance', 'category')
StatusDecisionOfHospitalization=_Column('StatusDecisionOfHospitalization', 'category')
StatusDecisionOfObservation =   _Column('StatusDecisionOfObservation', 'category')
NameOfHospital =                _Column('NameOfHospital', 'category')
DIC =                           _Column('DIC', 'bool')
MV =                            _Column('MV', 'bool')
TransferredToHospitalFromAnotherHospital=_Column('TransferredToHospitalFromAnotherHospital', 'bool')
TransferredToHospitalFromQuarantine =    _Column('TransferredToHospitalFromQuarantine', 'bool')
TestInformation =               _Column('TestInformation', 'category')
TypeOfPneumonia =               _Column('TypeOfPneumonia', 'category')
AntiviralTreatment =            _Column('AntiviralTreatment', 'bool')
ImmunosuppressantsDrugs =       _Column('ImmunosuppressantsDrugs', 'bool')
TreatmentHivInfectionDrugs =    _Column('TreatmentHivInfectionDrugs', 'bool')
AntiviralDrugs =                _Column('AntiviralDrugs', 'bool')
ECMO =                          _Column('ECMO', 'bool')
SaturationLevel =               _Column('SaturationLevel', 'int8')
ResultOfHospitalization =       _Column('ResultOfHospitalization', 'category')
ResultOfObservation =           _Column('ResultOfObservation', 'category')
ResultOfQuarantine =            _Column('ResultOfQuarantine', 'category')
ResultOfAftercare =             _Column('ResultOfAftercare', 'category')
ResultOfCT =                    _Column('ResultOfCT', 'category')
SeverityOfDisease =             _Column('SeverityOfDisease', 'category')
GroupOfRisk =                   _Column('GroupOfRisk', 'category')
Country =                       _Column('Country', 'category')
Longitude =                     _Column('Longitude', 'float64')
Latitude =                      _Column('Latitude', 'float64')


all_result = SimilarColumns('AllResult', [ResultOfHospitalization, ResultOfObservation, ResultOfQuarantine, ResultOfAftercare, ResultOfCT])
